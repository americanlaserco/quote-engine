"""Engraving calibration & measurement feedback loop.

Stores the actual measured engrave times of real jobs and re-fits the
engraving model from them, so accuracy improves as the shop logs more work.

Two files live in the project root:
  engrave_measurements.json — the raw log of measured jobs
  engrave_calibration.json  — the fitted result the quoting engine reads:
      {"global": {accel, job_overhead},
       "files":  {design_name: {speed: offset_seconds}},
       "stats":  {...}}

This mirrors the cut engine's calibration.py: a design that has been measured
gets a per-design offset and quotes near-exactly; unmeasured designs use the
fitted global model.
"""

import json
import time
from pathlib import Path

from laser_time_tool.engrave_planner import (
    _collect_edges, _scanline_spans,
    ENGRAVE_ACCEL_MM_S2, ENGRAVE_JOB_OVERHEAD_S, ENGRAVE_INTERVAL_MM,
)
from laser_time_tool.motion_planner import _single_move_time

_ROOT = Path(__file__).parent.parent
_MEAS_PATH = _ROOT / "engrave_measurements.json"
_CAL_PATH = _ROOT / "engrave_calibration.json"


def _norm(name) -> str:
    """Normalise a design name for matching: lowercase, no extension."""
    n = str(name).lower().strip()
    for ext in (".dxf", ".pdf", ".ai"):
        if n.endswith(ext):
            return n[:-len(ext)]
    return n


def compute_extents(paths, interval: float = ENGRAVE_INTERVAL_MM):
    """Return (extent_histogram, n_lines) for a set of engrave paths.

    extent_histogram maps str(rounded-mm scan-line extent) -> line count.
    This compact signature is all the re-fit needs — the original file is
    not required again.
    """
    edges = _collect_edges(paths)
    if not edges:
        return {}, 0
    ys = [p[1] for e in edges for p in e]
    y_min, y_max = min(ys), max(ys)
    hist = {}
    n = 0
    y = y_min + interval / 2.0
    while y < y_max:
        sp = _scanline_spans(edges, y)
        if sp:
            ext = sp[-1][1] - sp[0][0]
            k = str(int(round(ext)))
            hist[k] = hist.get(k, 0) + 1
            n += 1
        y += interval
    return hist, n


def _model_seconds(hist: dict, speed: float, accel: float,
                   job_overhead: float) -> float:
    """Engrave seconds predicted from an extent histogram (matches the engine)."""
    t = float(job_overhead)
    for ext_s, cnt in hist.items():
        t += cnt * _single_move_time(float(ext_s), speed, accel)
    return t


def _load_measurements() -> list:
    if _MEAS_PATH.exists():
        try:
            return json.loads(_MEAS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _load_calibration() -> dict:
    if _CAL_PATH.exists():
        try:
            return json.loads(_CAL_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def get_global_params() -> tuple:
    """Return (accel, job_overhead): fitted values if available, else defaults."""
    g = _load_calibration().get("global", {})
    return (float(g.get("accel", ENGRAVE_ACCEL_MM_S2)),
            float(g.get("job_overhead", ENGRAVE_JOB_OVERHEAD_S)))


def get_offset(name, speed: float) -> float:
    """Per-design correction (s), interpolated by speed. 0 for unmeasured designs."""
    entry = _load_calibration().get("files", {}).get(_norm(name))
    if not entry:
        return 0.0
    pts = sorted((float(s), float(v)) for s, v in entry.items())
    if not pts:
        return 0.0
    if len(pts) == 1 or speed <= pts[0][0]:
        return pts[0][1]
    if speed >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        s1, v1 = pts[i]
        s2, v2 = pts[i + 1]
        if s1 <= speed <= s2:
            return v1 + (v2 - v1) * (speed - s1) / (s2 - s1)
    return pts[-1][1]


# Unseen designs use the global model, which cross-validation showed runs
# ~1-2% low on average. This small factor cancels that under-bias so an
# un-measured estimate lands slightly over rather than under, while staying
# close to the true time. Measured designs keep their exact per-design
# offset and are NOT marked up.
ENGRAVE_SAFETY_FACTOR = 1.03


def is_measured_design(name) -> bool:
    """True if this design has real measured calibration data on file."""
    return _norm(name) in _load_calibration().get("files", {})


def apply_calibration(raw_seconds: float, name, speed: float) -> float:
    """Apply calibration to a raw engrave estimate.

    A design with measured data gets its exact per-design offset. A design
    with no measured data is biased up by ENGRAVE_SAFETY_FACTOR so it errs
    high (over-quote) rather than low.
    """
    files = _load_calibration().get("files", {})
    if _norm(name) in files:
        return max(0.0, raw_seconds + get_offset(name, speed))
    return max(0.0, raw_seconds * ENGRAVE_SAFETY_FACTOR)


def _fit_global(measurements: list) -> tuple:
    """Coarse-to-fine grid fit of (accel, job_overhead) over all measurements."""
    usable = [m for m in measurements if m.get("actual", 0) > 0 and m.get("extents")]
    if not usable:
        return ENGRAVE_ACCEL_MM_S2, ENGRAVE_JOB_OVERHEAD_S

    def err(accel, joh):
        s = 0.0
        for m in usable:
            pred = _model_seconds(m["extents"], m["speed"], accel, joh)
            s += abs(pred - m["actual"]) / m["actual"]
        return s / len(usable)

    best = None
    for accel in range(400, 4001, 200):
        for joh in range(0, 61, 4):
            e = err(accel, joh)
            if best is None or e < best[0]:
                best = (e, accel, joh)
    _, accel, joh = best
    for _ in range(3):
        cur = best
        for da in (-150, -50, -20, 0, 20, 50, 150):
            for dj in (-3, -1, 0, 1, 3):
                a2 = max(100, accel + da)
                j2 = max(0, joh + dj)
                e = err(a2, j2)
                if e < cur[0]:
                    cur = (e, a2, j2)
        best = cur
        _, accel, joh = best
    return float(accel), float(joh)


def recalibrate() -> dict:
    """Re-fit from all measurements, write engrave_calibration.json, return stats."""
    meas = _load_measurements()
    usable = [m for m in meas if m.get("actual", 0) > 0 and m.get("extents")]
    if not usable:
        return {"n_measurements": 0, "model_err_pct": None}

    accel, joh = _fit_global(usable)

    files = {}
    raw_errs = []
    for m in usable:
        pred = _model_seconds(m["extents"], m["speed"], accel, joh)
        raw_errs.append(abs(pred - m["actual"]) / m["actual"])
        offset = m["actual"] - pred
        files.setdefault(_norm(m["name"]), {})[str(int(round(m["speed"])))] = round(offset, 1)

    stats = {
        "n_measurements": len(usable),
        "n_designs": len(files),
        "model_err_pct": round(100 * sum(raw_errs) / len(raw_errs), 2),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cal = {"global": {"accel": accel, "job_overhead": joh},
           "files": files, "stats": stats}
    _CAL_PATH.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    return stats


def record_measurement(name, speed, interval, actual_seconds,
                       extents, n_lines) -> dict:
    """Append (or replace) a measured job, then re-fit. Returns updated stats."""
    meas = _load_measurements()
    nm = _norm(name)
    # Replace any existing entry for the same design + speed.
    meas = [m for m in meas
            if not (_norm(m["name"]) == nm and abs(m["speed"] - speed) < 1.0)]
    meas.append({
        "name": nm, "speed": float(speed), "interval": float(interval),
        "actual": float(actual_seconds), "extents": extents,
        "n_lines": int(n_lines), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    _MEAS_PATH.write_text(json.dumps(meas, indent=2), encoding="utf-8")
    return recalibrate()
