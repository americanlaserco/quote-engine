"""Bench-calibrator runner — autonomous Ruida bench-controller calibration.

Drives a bench Ruida controller (no laser attached) through a folder of
DXF/PDF/AI files at multiple cut speeds, asks the controller to predict
each file's run time via 0xE8 0x04 / memory 0x07 0x10, and records the
results into bench_results.json. Resumable, runs in a background thread
so the Flask server can serve a live status UI.

Designed to be driven from server.py via /api/bench-calibrate/* endpoints.
"""

import json
import os
import threading
import time
from pathlib import Path

# -------- Discovery: find a connected Ruida -----------------------------------

def discover_controller(timeout: float = 2.0):
    """Try to find a connected Ruida controller.

    Returns a dict describing the connection on success:
        {'kind': 'serial', 'port': 'COM4', 'card_id': '0x65106510'}
        {'kind': 'udp',    'ip':   '192.168.1.100', 'card_id': '...'}
    Returns None if nothing found.
    """
    # 1) Try serial ports (USB-attached controllers)
    try:
        import serial.tools.list_ports
        from laser_time_tool.ruida_connection import RuidaSerialConnection
        for p in serial.tools.list_ports.comports():
            try:
                conn = RuidaSerialConnection(port=p.device, timeout=timeout)
                if conn.connect():
                    cid = conn.query_value(b"\x05\x7E")
                    conn.close()
                    if cid is not None:
                        return {'kind': 'serial', 'port': p.device,
                                'card_id': hex(cid), 'desc': p.description}
                conn.close()
            except Exception:
                continue
    except Exception:
        pass

    # 2) Try common UDP addresses on local network
    try:
        from laser_time_tool.ruida_connection import RuidaConnection
        import socket
        # Resolve local subnet and probe .100, .101, .200 — common Ruida defaults
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if '.' in local_ip:
            parts = local_ip.split('.')
            subnet = '.'.join(parts[:3])
            for last in (100, 101, 200, 1, 2):
                ip = f"{subnet}.{last}"
                try:
                    conn = RuidaConnection(ip=ip, timeout=0.5)
                    if conn.connect():
                        cid = conn.get_card_id()
                        conn.close()
                        if cid is not None:
                            return {'kind': 'udp', 'ip': ip, 'card_id': hex(cid)}
                    conn.close()
                except Exception:
                    continue
    except Exception:
        pass

    return None


# -------- Calibration job runner ---------------------------------------------

class CalibrationJob:
    """Background-thread calibration runner.

    Walks a folder of DXF/PDF/AI files, builds an .rd file for each at each
    requested speed, asks the controller to predict the time, and records.
    Status is exposed via .snapshot() for the web UI to poll.
    """

    def __init__(self,
                 results_path: Path,
                 folders: list,
                 speeds: list,
                 connection_info: dict,
                 force: bool = False):
        self.results_path = Path(results_path)
        self.folders = [Path(f) for f in folders]
        self.speeds = list(speeds)
        self.conn_info = connection_info
        self.force = force

        self.thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # Public state
        self.status = "idle"            # idle | running | finished | stopped | error
        self.error_message = None
        self.queue = []                  # list of (path, speed) yet to do
        self.done = []                   # list of result dicts already recorded
        self.errors = []                 # list of dicts {file, speed, reason, ts}
        self.current = None              # currently-processing (path, speed)
        self.t_start = None
        self.t_end = None

    # ---- public state ----
    def snapshot(self):
        with self._lock:
            return {
                'status': self.status,
                'error_message': self.error_message,
                'connection': self.conn_info,
                'progress_done': len(self.done),
                'progress_total': len(self.done) + len(self.queue) + (1 if self.current else 0),
                'current': self.current,
                'recent_done': self.done[-20:],
                'recent_errors': self.errors[-10:],
                'all_done_count': len(self.done),
                'all_error_count': len(self.errors),
                'elapsed_s': (time.time() - self.t_start) if self.t_start else 0,
                'finished_at': self.t_end,
            }

    def all_results(self):
        with self._lock:
            return list(self.done)

    # ---- lifecycle ----
    def start(self):
        if self.thread and self.thread.is_alive():
            return False
        self._stop.clear()
        self.status = "running"
        self.error_message = None
        self.t_start = time.time()
        self.t_end = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=10)
        with self._lock:
            if self.status == "running":
                self.status = "stopped"
                self.t_end = time.time()

    # ---- main loop ----
    def _enumerate_files(self):
        """Find every DXF/PDF/AI in the picked folders and pair with each speed."""
        exts = (".dxf", ".pdf", ".ai")
        files = []
        for folder in self.folders:
            if not folder.exists():
                continue
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in exts and not p.name.startswith('.'):
                    files.append(p)
        return [(f, s) for f in files for s in self.speeds]

    def _load_existing(self):
        """Load already-recorded results so we can resume."""
        if self.results_path.exists() and not self.force:
            try:
                data = json.loads(self.results_path.read_text(encoding='utf-8'))
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        return []

    def _save_results(self):
        """Persist current done-list to bench_results.json."""
        try:
            tmp = self.results_path.with_suffix(self.results_path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.done, indent=2), encoding='utf-8')
            tmp.replace(self.results_path)
        except Exception as e:
            self.errors.append({'file': '(save)', 'speed': 0,
                               'reason': f'save failed: {e}', 'ts': time.time()})

    def _open_controller(self):
        """Construct a controller connection from self.conn_info."""
        kind = self.conn_info.get('kind')
        if kind == 'serial':
            from laser_time_tool.ruida_connection import RuidaSerialConnection
            conn = RuidaSerialConnection(port=self.conn_info['port'])
            if not conn.connect():
                return None
            return conn
        if kind == 'udp':
            from laser_time_tool.ruida_connection import RuidaConnection
            conn = RuidaConnection(ip=self.conn_info['ip'])
            if not conn.connect():
                return None
            return conn
        if kind == 'mock':
            return _MockConnection()
        return None

    def _build_rd(self, path: Path, speed: float):
        """Parse the source file, build an .rd buffer at given speed."""
        from laser_time_tool.rd_file_builder import build_rd_from_paths
        ext = path.suffix.lower()
        if ext == '.dxf':
            from laser_time_tool.dxf_parser import parse_dxf
            cp = parse_dxf(str(path), {c: speed for c in range(1, 256)})
        else:
            from server import parse_vector
            cp = parse_vector(str(path), {c: speed for c in range(1, 256)})
        if not cp:
            return None
        csm = {a: speed for a in cp}
        return build_rd_from_paths(cp, csm)

    def _run(self):
        try:
            # Build/restore queue
            existing = self._load_existing()
            with self._lock:
                self.done = existing
                done_keys = {(r['file'], r['ext'], float(r['speed'])) for r in existing}
                all_pairs = self._enumerate_files()
                self.queue = [
                    (str(p), float(s)) for p, s in all_pairs
                    if (p.stem, p.suffix.lower().lstrip('.'), float(s)) not in done_keys
                ]

            conn = self._open_controller()
            if conn is None:
                with self._lock:
                    self.status = "error"
                    self.error_message = "Could not open controller connection"
                    self.t_end = time.time()
                return

            try:
                # Process the queue
                while True:
                    if self._stop.is_set():
                        break
                    with self._lock:
                        if not self.queue:
                            break
                        path_str, speed = self.queue.pop(0)
                        self.current = {'file': Path(path_str).stem,
                                       'ext': Path(path_str).suffix.lower().lstrip('.'),
                                       'speed': speed, 'path': path_str}
                    self._process_one(conn, Path(path_str), speed)
                    # Persist every 10 results so progress survives crashes
                    if len(self.done) % 10 == 0:
                        self._save_results()

                self._save_results()
                with self._lock:
                    self.current = None
                    self.status = "finished" if not self._stop.is_set() else "stopped"
                    self.t_end = time.time()
            finally:
                try: conn.close()
                except Exception: pass
        except Exception as e:
            with self._lock:
                self.status = "error"
                self.error_message = f"{type(e).__name__}: {e}"
                self.t_end = time.time()

    def _process_one(self, conn, path: Path, speed: float):
        """Calibrate one (file, speed) and append to self.done or self.errors."""
        stem = path.stem; ext = path.suffix.lower().lstrip('.')
        try:
            rd_bytes = self._build_rd(path, speed)
            if not rd_bytes:
                with self._lock:
                    self.errors.append({'file': stem, 'ext': ext, 'speed': speed,
                                       'reason': 'no cuttable paths', 'ts': time.time()})
                    self.current = None
                return
            predicted = conn.predict_job_time(rd_bytes, filename=f"{stem}.rd")
            ts = time.time()
            if predicted is None or predicted <= 0:
                with self._lock:
                    self.errors.append({'file': stem, 'ext': ext, 'speed': speed,
                                       'reason': 'controller returned no time', 'ts': ts})
                    self.current = None
                return
            with self._lock:
                self.done.append({
                    'file': stem, 'ext': ext, 'speed': speed,
                    'controller_predicted_s': round(predicted, 2),
                    'ts': ts,
                })
                self.current = None
        except Exception as e:
            with self._lock:
                self.errors.append({'file': stem, 'ext': ext, 'speed': speed,
                                   'reason': f'{type(e).__name__}: {e}', 'ts': time.time()})
                self.current = None


# -------- Mock controller for sandbox testing --------------------------------

class _MockConnection:
    """Simulates a Ruida bench controller. Used for sandbox dry-runs only.

    Returns a deterministic-but-plausible predicted time based on the .rd
    file size and a small added overhead, so tests can run end-to-end
    without a real controller.
    """
    def __init__(self):
        self.last_rd = None
    def predict_job_time(self, rd_data: bytes, filename: str = "job.rd",
                         doc_index: int = 1):
        self.last_rd = rd_data
        # Plausible: 0.01 s per byte + 5 s base + small jitter from filename
        base = 5.0
        per_byte = 0.001
        return round(base + len(rd_data) * per_byte
                     + (hash(filename) % 7), 2)
    def close(self):
        pass


# -------- Merge results into calibration_table.json --------------------------

def merge_into_calibration(results_path: Path, table_path: Path):
    """Take bench_results.json and merge controller-predicted times into the
    main calibration_table.json. Each entry becomes an offset = predicted - raw,
    keyed by (filename_stem, speed) just like the existing table.

    Returns a dict with counts of {added, updated, skipped}.
    """
    import sys
    sys.path.insert(0, '.')
    from laser_time_tool.dxf_parser import parse_dxf, path_length
    from laser_time_tool.motion_planner import estimate_time_offline

    results = json.loads(Path(results_path).read_text(encoding='utf-8'))
    table = {}
    if Path(table_path).exists():
        table = json.loads(Path(table_path).read_text(encoding='utf-8'))

    added = updated = skipped = 0
    # Find source DXFs for each stem
    search_dirs = [Path('test files'), Path('../convertedfiles'),
                  Path('../final test files copy'),
                  Path('/sessions/hopeful-loving-lamport/mnt/convertedfiles'),
                  Path('/sessions/hopeful-loving-lamport/mnt/final test files copy')]

    for r in results:
        stem = r['file']; speed = float(r['speed'])
        predicted = float(r['controller_predicted_s'])
        # Need to compute raw at this speed to derive an offset
        dxf = None
        for d in search_dirs:
            p = d / f"{stem}.dxf"
            if p.exists(): dxf = p; break
        if not dxf:
            skipped += 1; continue
        try:
            csm = {c: speed for c in range(1, 256)}
            cp = parse_dxf(str(dxf), csm)
            raw = estimate_time_offline(cp, csm)
            offset = round(predicted - raw, 1)
            entry = table.get(stem, {})
            speed_key = str(int(speed))
            if speed_key in entry:
                updated += 1
            else:
                added += 1
            entry[speed_key] = offset
            table[stem] = entry
        except Exception:
            skipped += 1

    Path(table_path).write_text(json.dumps(table, indent=2, sort_keys=True), encoding='utf-8')
    return {'added': added, 'updated': updated, 'skipped': skipped,
            'total_entries': sum(len(v) for v in table.values())}
