"""Ruida laser controller communication via UDP (Ethernet).

Supports RDC6442, RDC6445, RDC6332 and similar Ruida controllers.
Protocol reverse-engineered from meerk40t and jnweiger/ruida-laser projects.

Two UDP interfaces:
  - Port 50200 (data): file upload, DA queries, bulk commands (all swizzled)
  - Port 50207 (realtime): jog, start/stop/pause (NOT swizzled)
"""

import socket
import struct
import time

# ---------------------------------------------------------------------------
# Swizzle (scramble/unscramble) — Ruida XOR encoding
# ---------------------------------------------------------------------------

MAGIC = 0x88  # Standard for RDC644XG, RDC644XS, RDC320, RDC633X, RDC654XG


def _swizzle_byte(b: int, magic: int = MAGIC) -> int:
    b ^= (b >> 7) & 0xFF
    b ^= (b << 7) & 0xFF
    b ^= (b >> 7) & 0xFF
    b ^= magic
    return (b + 1) & 0xFF


def _unswizzle_byte(b: int, magic: int = MAGIC) -> int:
    b = (b - 1) & 0xFF
    b ^= magic
    b ^= (b >> 7) & 0xFF
    b ^= (b << 7) & 0xFF
    b ^= (b >> 7) & 0xFF
    return b


# Build lookup tables for speed
SWIZZLE_LUT = [_swizzle_byte(i) for i in range(256)]
UNSWIZZLE_LUT = [_unswizzle_byte(i) for i in range(256)]


def swizzle(data: bytes) -> bytes:
    """Scramble bytes for sending to Ruida controller."""
    return bytes([SWIZZLE_LUT[b] for b in data])


def unswizzle(data: bytes) -> bytes:
    """Unscramble bytes received from Ruida controller."""
    return bytes([UNSWIZZLE_LUT[b] for b in data])


# ---------------------------------------------------------------------------
# Value encoding — Ruida uses 7-bit packing (MSB=0 for data bytes)
# ---------------------------------------------------------------------------

def encode32(v: int) -> bytes:
    """Encode integer into 5-byte 7-bit packed format (35 bits effective)."""
    v = int(v) & 0x7FFFFFFFF  # 35-bit mask
    return bytes([
        (v >> 28) & 0x7F,
        (v >> 21) & 0x7F,
        (v >> 14) & 0x7F,
        (v >> 7) & 0x7F,
        v & 0x7F,
    ])


def decode35(data: bytes) -> int:
    """Decode 5-byte 7-bit packed value back to integer."""
    v = 0
    for b in data[:5]:
        v = (v << 7) | (b & 0x7F)
    return v


def encode_coord(mm: float) -> bytes:
    """Encode a coordinate in mm to 5-byte format (stored as micrometers)."""
    um = int(round(mm * 1000.0))
    if um < 0:
        um = 0
    return encode32(um)


def encode_speed(mm_s: float) -> bytes:
    """Encode speed in mm/s to 5-byte format (stored as μm/s)."""
    return encode32(int(round(mm_s * 1000.0)))


def encode_power(percent: float) -> bytes:
    """Encode power 0-100% to 2-byte 14-bit format."""
    val = int(percent * 16383.0 / 100.0)
    val = max(0, min(16383, val))
    return bytes([(val >> 7) & 0x7F, val & 0x7F])


def encode_color(r: int, g: int, b: int) -> bytes:
    """Encode RGB color to 5-byte format (BGR packed)."""
    return encode32((b << 16) | (g << 8) | r)


# ---------------------------------------------------------------------------
# Memory addresses for DA 00 queries
# ---------------------------------------------------------------------------

MEM_MACHINE_STATUS = b"\x04\x00"
MEM_CURRENT_X = b"\x04\x21"
MEM_CURRENT_Y = b"\x04\x31"
MEM_BED_SIZE_X = b"\x00\x26"
MEM_BED_SIZE_Y = b"\x00\x36"
MEM_CARD_ID = b"\x05\x7E"
MEM_TOTAL_OPEN_TIME = b"\x04\x01"
MEM_TOTAL_WORK_TIME = b"\x04\x02"
MEM_TOTAL_WORK_COUNT = b"\x04\x03"
MEM_PREV_WORK_TIME = b"\x04\x08"   # Duration of last completed job
MEM_TOTAL_LASER_TIME = b"\x04\x11"
MEM_DOC_COUNT = b"\x04\x05"
MEM_FILE_TOTAL_LEN = b"\x06\x20"   # Total length of current file (progress tracking)
MEM_FILE_PROGRESS = b"\x06\x21"    # Current progress through file
MEM_FILE_1_TIME = b"\x03\x91"      # Time for file 1 to run


# ---------------------------------------------------------------------------
# RuidaConnection — UDP communication
# ---------------------------------------------------------------------------

MEM_DOCUMENT_TIME = b"\x07\x10"   # 0x390 — controller-calculated time for last-selected document (seconds)

class RuidaConnectionError(Exception):
    """Connection or communication failure with Ruida controller."""
    pass


class RuidaConnection:
    """UDP connection to a Ruida laser controller."""

    DATA_PORT = 50200       # Controller's data port
    REALTIME_PORT = 50207   # Controller's realtime port
    LOCAL_DATA_PORT = 40200
    LOCAL_RT_PORT = 40207
    MTU = 1000              # Safe chunk size for UDP payload
    ACK_SWIZZLED = 0xC6     # Swizzled ACK byte from controller

    def __init__(self, ip: str = "192.168.1.100", timeout: float = 3.0):
        self.ip = ip
        self.timeout = timeout
        self.data_sock = None
        self.rt_sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def connect(self) -> bool:
        """Open UDP sockets and verify controller responds."""
        # Data socket (swizzled protocol)
        self.data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.data_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.data_sock.settimeout(self.timeout)
        try:
            self.data_sock.bind(("", self.LOCAL_DATA_PORT))
        except OSError:
            # Port already in use — try without binding to specific port
            self.data_sock.bind(("", 0))

        # Realtime socket (unswizzled protocol)
        self.rt_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rt_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rt_sock.settimeout(self.timeout)
        try:
            self.rt_sock.bind(("", self.LOCAL_RT_PORT))
        except OSError:
            self.rt_sock.bind(("", 0))

        # Handshake — send ENQ (0xCE) and wait for ACK
        return self._handshake()

    def _handshake(self) -> bool:
        """Send ENQ and wait for ACK to verify controller is reachable."""
        enq = b"\xCE"
        pkt = self._make_packet(enq)
        for attempt in range(3):
            try:
                self.data_sock.sendto(pkt, (self.ip, self.DATA_PORT))
                data, addr = self.data_sock.recvfrom(1024)
                if len(data) > 0 and data[0] == self.ACK_SWIZZLED:
                    return True
                # Also check unswizzled
                if len(data) > 0:
                    reply = unswizzle(data)
                    if reply[0:1] == b"\xCC":  # ACK unswizzled
                        return True
            except socket.timeout:
                continue
        return False

    def close(self):
        """Close UDP sockets."""
        if self.data_sock:
            self.data_sock.close()
            self.data_sock = None
        if self.rt_sock:
            self.rt_sock.close()
            self.rt_sock = None

    def _make_packet(self, raw_data: bytes) -> bytes:
        """Swizzle data and prepend 2-byte big-endian checksum."""
        scrambled = swizzle(raw_data)
        cs = sum(scrambled) & 0xFFFF
        return struct.pack(">H", cs) + scrambled

    def _send_packet(self, raw_data: bytes, retry_on_nak: bool = False) -> bytes | None:
        """Send a swizzled packet and wait for ACK. Returns response bytes or None."""
        pkt = self._make_packet(raw_data)
        retry_delay = 0.2
        max_retries = 5 if retry_on_nak else 1

        for attempt in range(max_retries):
            try:
                self.data_sock.sendto(pkt, (self.ip, self.DATA_PORT))
                data, addr = self.data_sock.recvfrom(4096)
                if len(data) == 0:
                    continue

                # Check for ACK (swizzled 0xCC = 0xC6)
                if data[0] == self.ACK_SWIZZLED:
                    return b"\xCC"  # Return canonical ACK

                # Check for NAK
                unscrambled = unswizzle(data)
                if unscrambled[0:1] == b"\xCF":  # NAK
                    if retry_on_nak:
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 5.0)
                        continue
                    return None

                # Data response — return unswizzled
                return unscrambled

            except socket.timeout:
                if retry_on_nak and attempt < max_retries - 1:
                    continue
                return None

        return None

    def upload_rd_file(self, rd_data: bytes, filename: str = "job.rd") -> bool:
        """Upload scrambled .rd file data to the controller.

        The rd_data should already be scrambled (output of RdFileBuilder.build()).
        """
        # Set filename: E8 02 E7 01 <filename null-terminated>
        name_cmd = b"\xE8\x02\xE7\x01" + filename.encode("ascii") + b"\x00"
        result = self._send_packet(name_cmd, retry_on_nak=True)
        if result is None:
            return False

        # Upload file data in chunks
        offset = 0
        total = len(rd_data)
        first_chunk = True

        while offset < total:
            chunk_size = min(self.MTU, total - offset)
            chunk = rd_data[offset:offset + chunk_size]

            # For already-scrambled data, compute checksum directly and send
            cs = sum(chunk) & 0xFFFF
            pkt = struct.pack(">H", cs) + chunk
            try:
                self.data_sock.sendto(pkt, (self.ip, self.DATA_PORT))
                resp, _ = self.data_sock.recvfrom(1024)

                if len(resp) > 0 and resp[0] == self.ACK_SWIZZLED:
                    offset += chunk_size
                    first_chunk = False
                    continue

                # NAK — retry first chunk, abort on later chunks
                if first_chunk:
                    time.sleep(0.2)
                    continue
                else:
                    return False

            except socket.timeout:
                if first_chunk:
                    continue
                return False

        return True

    def query_value(self, mem_addr: bytes) -> int | None:
        """Send DA 00 query and parse DA 01 response.

        The Ruida sends TWO packets in response:
          1. A 1-byte ACK (0xC6 swizzled)
          2. A 9-byte DA 01 data response

        Returns decoded value or None.
        """
        cmd = b"\xDA\x00" + mem_addr
        pkt = self._make_packet(cmd)

        old_timeout = self.data_sock.gettimeout()
        self.data_sock.settimeout(min(self.timeout, 2.0))

        try:
            self.data_sock.sendto(pkt, (self.ip, self.DATA_PORT))

            # Read up to 3 packets (ACK + data, with possible retransmits)
            for _ in range(3):
                try:
                    data, _ = self.data_sock.recvfrom(4096)
                except socket.timeout:
                    break

                if len(data) < 2:
                    # Likely just an ACK, continue reading
                    continue

                raw = unswizzle(data)

                # Look for DA 01 response pattern
                for i in range(len(raw) - 8):
                    if raw[i] == 0xDA and raw[i + 1] == 0x01:
                        if i + 9 <= len(raw):
                            value = decode35(raw[i + 4:i + 9])
                            self.data_sock.settimeout(old_timeout)
                            return value

            self.data_sock.settimeout(old_timeout)
            return None

        except socket.timeout:
            self.data_sock.settimeout(old_timeout)
            return None

    def get_machine_status(self) -> dict | None:
        """Query controller for machine status."""
        status = self.query_value(MEM_MACHINE_STATUS)
        if status is None:
            return None

        return {
            "raw": status,
            "running": bool(status & 0x01),
            "idle": status in (0x16, 22),
            "paused": status in (0x17, 23),
        }

    def get_position(self) -> tuple[float, float] | None:
        """Get current head position in mm."""
        x = self.query_value(MEM_CURRENT_X)
        y = self.query_value(MEM_CURRENT_Y)
        if x is None or y is None:
            return None
        return (x / 1000.0, y / 1000.0)

    def get_bed_size(self) -> tuple[float, float] | None:
        """Get bed size in mm."""
        x = self.query_value(MEM_BED_SIZE_X)
        y = self.query_value(MEM_BED_SIZE_Y)
        if x is None or y is None:
            return None
        return (x / 1000.0, y / 1000.0)

    def get_card_id(self) -> int | None:
        """Get controller card ID."""
        return self.query_value(MEM_CARD_ID)

    def get_total_work_time(self) -> float | None:
        """Get total accumulated work time in seconds."""
        val = self.query_value(MEM_TOTAL_WORK_TIME)
        return float(val) if val is not None else None

    def get_file_run_time(self, file_num: int = 1) -> float | None:
        """Query estimated/actual run time for a stored file.

        Memory addresses 0x0391-0x0420 store per-file run times.
        file_num: 1-based file index.
        """
        addr_val = 0x0390 + file_num
        addr = bytes([(addr_val >> 7) & 0x7F, addr_val & 0x7F])
        val = self.query_value(addr)
        return float(val) if val is not None else None

    def get_doc_count(self) -> int | None:
        """Get number of documents stored on controller."""
        return self.query_value(MEM_DOC_COUNT)

    def get_prev_work_time(self) -> float | None:
        """Get duration of the last completed job in seconds."""
        val = self.query_value(MEM_PREV_WORK_TIME)
        return float(val) if val is not None else None

    def select_document(self, index: int = 1) -> bool:
        """Select which uploaded document the next command should target.

        Most controllers accept 0xE8 0x03 <index>; some firmware variants
        ignore it and operate on whatever's currently loaded. Safe to call
        even if unnecessary.
        """
        result = self._send_packet(b"\xE8\x03" + bytes([index & 0xFF]))
        return result is not None

    def calculate_document_time(self) -> bool:
        """Ask the controller to run its internal time prediction on the
        currently-loaded document.

        After this command, the calculated value should be readable from
        the Document Time memory address (0x07 0x10). The calculation is
        typically fast (<1 second) — far less than actually running the job.
        """
        result = self._send_packet(b"\xE8\x04")
        return result is not None

    def get_document_time(self) -> float | None:
        """Read the controller's predicted time for the currently-loaded
        document, in seconds. Returns None on failure."""
        val = self.query_value(MEM_DOCUMENT_TIME)
        return float(val) if val is not None else None

    def predict_job_time(self, rd_data: bytes,
                        filename: str = "job.rd",
                        doc_index: int = 1) -> float | None:
        """End-to-end: upload an .rd file, ask the controller to calculate
        its time, and return the predicted seconds. None on failure.

        This is the workhorse for autonomous bench calibration."""
        if not self.upload_rd_file(rd_data, filename=filename):
            return None
        # Best-effort document selection (some firmware ignores it).
        self.select_document(doc_index)
        if not self.calculate_document_time():
            return None
        import time
        # Give the controller a moment to finish its internal calculation.
        time.sleep(0.5)
        return self.get_document_time()


    def get_file_progress(self) -> tuple[int | None, int | None]:
        """Get file total length and current progress (for job % tracking)."""
        total = self.query_value(MEM_FILE_TOTAL_LEN)
        progress = self.query_value(MEM_FILE_PROGRESS)
        return (total, progress)

    def send_raw_data_cmd(self, raw_cmd: bytes) -> bytes | None:
        """Send a raw command on the data port (swizzled) and return response."""
        return self._send_packet(raw_cmd)

    def query_raw(self, mem_hi: int, mem_lo: int) -> int | None:
        """Query an arbitrary DA 00 address given high and low bytes."""
        addr = bytes([mem_hi & 0x7F, mem_lo & 0x7F])
        return self.query_value(addr)

    # -----------------------------------------------------------------------
    # Realtime commands (port 50207, NOT swizzled)
    # -----------------------------------------------------------------------

    def _send_realtime(self, data: bytes) -> bool:
        """Send a realtime command (not swizzled) to port 50207."""
        if not self.rt_sock:
            return False
        try:
            self.rt_sock.sendto(data, (self.ip, self.REALTIME_PORT))
            return True
        except Exception:
            return False

    def start_job(self) -> bool:
        """Start/resume the loaded job (via data port, swizzled)."""
        result = self._send_packet(b"\xD8\x00")
        return result is not None

    def stop_job(self) -> bool:
        """Stop the current job (via data port, swizzled)."""
        result = self._send_packet(b"\xD8\x01")
        return result is not None

    def pause_job(self) -> bool:
        """Pause the current job (via data port, swizzled)."""
        result = self._send_packet(b"\xD8\x02")
        return result is not None

    def resume_job(self) -> bool:
        """Resume a paused job (via data port, swizzled)."""
        result = self._send_packet(b"\xD8\x03")
        return result is not None


# ---------------------------------------------------------------------------
# RuidaSerialConnection — USB serial communication (COM port)
# ---------------------------------------------------------------------------

class RuidaSerialConnection:
    """Serial/USB connection to a Ruida laser controller.

    Some Ruida controllers expose a serial port via CH340/FTDI chip.
    The protocol uses the same scrambled byte commands as UDP but
    without the 2-byte checksum framing.
    """

    def __init__(self, port: str = "COM4", baudrate: int = 921600, timeout: float = 2.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def connect(self) -> bool:
        """Open serial connection and verify controller responds."""
        import serial

        try:
            self.ser = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
        except Exception as e:
            # Try common Ruida baudrates
            for baud in [115200, 57600, 9600]:
                try:
                    self.ser = serial.Serial(
                        self.port, baud,
                        timeout=self.timeout,
                        write_timeout=self.timeout,
                    )
                    break
                except Exception:
                    continue
            else:
                return False

        import time
        time.sleep(0.5)
        self.ser.reset_input_buffer()

        # Try handshake — send ENQ (0xCE scrambled)
        return self._handshake()

    def _handshake(self) -> bool:
        """Send ENQ and check for ACK over serial."""
        enq = swizzle(b"\xCE")
        for attempt in range(3):
            try:
                self.ser.write(enq)
                self.ser.flush()
                import time
                time.sleep(0.1)

                if self.ser.in_waiting > 0:
                    resp = self.ser.read(self.ser.in_waiting)
                    if resp:
                        # Check for ACK (swizzled 0xCC = 0xC6)
                        if 0xC6 in resp:
                            return True
                        # Try unswizzled check
                        for b in resp:
                            if _unswizzle_byte(b) == 0xCC:
                                return True
            except Exception:
                continue

        # Some Ruida controllers don't ACK the ENQ over serial but still
        # respond to data queries. Try a DA query (get card ID) as fallback.
        try:
            resp = self._send_command(b"\xDA\x00\x00\x04")
            if resp and len(resp) >= 1:
                return True
        except Exception:
            pass
        return False

    def close(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ser = None

    def _send_command(self, raw_data: bytes) -> bytes:
        """Send scrambled command and read response."""
        scrambled = swizzle(raw_data)
        try:
            self.ser.write(scrambled)
            self.ser.flush()

            import time
            time.sleep(0.1)

            # Read whatever is available
            resp = b""
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if self.ser.in_waiting > 0:
                    resp += self.ser.read(self.ser.in_waiting)
                    time.sleep(0.05)
                elif resp:
                    break
                else:
                    time.sleep(0.05)

            return unswizzle(resp) if resp else b""
        except Exception:
            return b""

    def upload_rd_file(self, rd_data: bytes, filename: str = "job.rd") -> bool:
        """Upload scrambled .rd file data over serial.

        The rd_data should already be scrambled.
        """
        # Set filename
        name_cmd = swizzle(b"\xE8\x02\xE7\x01" + filename.encode("ascii") + b"\x00")
        try:
            self.ser.write(name_cmd)
            self.ser.flush()
            import time
            time.sleep(0.2)

            # Flush response
            if self.ser.in_waiting:
                self.ser.read(self.ser.in_waiting)

            # Send file data in chunks
            chunk_size = 512
            offset = 0
            while offset < len(rd_data):
                chunk = rd_data[offset:offset + chunk_size]
                self.ser.write(chunk)
                self.ser.flush()
                time.sleep(0.01)
                offset += len(chunk)

            time.sleep(0.5)
            if self.ser.in_waiting:
                self.ser.read(self.ser.in_waiting)

            return True
        except Exception:
            return False
    def select_document(self, index: int = 1) -> bool:
        """Select which uploaded document the next command should target."""
        resp = self._send_command(b"\xE8\x03" + bytes([index & 0xFF]))
        return resp is not None

    def calculate_document_time(self) -> bool:
        """Ask the controller to run its internal time prediction on the
        currently-loaded document (the bench-calibration workhorse)."""
        resp = self._send_command(b"\xE8\x04")
        return resp is not None

    def query_value(self, mem_addr: bytes) -> int | None:
        """Send DA 00 query over serial and parse DA 01 response."""
        cmd = b"\xDA\x00" + mem_addr
        raw = self._send_command(cmd)
        if not raw: return None
        # Look for DA 01 pattern in unswizzled response
        for i in range(len(raw) - 8):
            if raw[i] == 0xDA and raw[i + 1] == 0x01:
                if i + 9 <= len(raw):
                    return decode35(raw[i + 4:i + 9])
        return None

    def get_document_time(self) -> float | None:
        """Read controller's predicted document time in seconds."""
        val = self.query_value(MEM_DOCUMENT_TIME)
        return float(val) if val is not None else None

    def predict_job_time(self, rd_data: bytes,
                        filename: str = "job.rd",
                        doc_index: int = 1) -> float | None:
        """End-to-end bench prediction over serial."""
        if not self.upload_rd_file(rd_data, filename=filename):
            return None
        self.select_document(doc_index)
        if not self.calculate_document_time():
            return None
        import time
        time.sleep(0.5)
        return self.get_document_time()


    def query_value(self, mem_addr: bytes) -> int | None:
        """Send DA 00 query and parse response."""
        cmd = b"\xDA\x00" + mem_addr
        resp = self._send_command(cmd)

        if len(resp) < 9:
            return None

        # Look for DA 01 response
        for i in range(len(resp) - 6):
            if resp[i] == 0xDA and resp[i + 1] == 0x01:
                if i + 9 <= len(resp):
                    return decode35(resp[i + 4:i + 9])
        return None

    def get_machine_status(self) -> dict | None:
        status = self.query_value(MEM_MACHINE_STATUS)
        if status is None:
            return None
        return {
            "raw": status,
            "running": bool(status & 0x01),
            "idle": status in (0x16, 22),
            "paused": status in (0x17, 23),
        }

    def get_position(self) -> tuple[float, float] | None:
        x = self.query_value(MEM_CURRENT_X)
        y = self.query_value(MEM_CURRENT_Y)
        if x is None or y is None:
            return None
        return (x / 1000.0, y / 1000.0)

    def get_bed_size(self) -> tuple[float, float] | None:
        x = self.query_value(MEM_BED_SIZE_X)
        y = self.query_value(MEM_BED_SIZE_Y)
        if x is None or y is None:
            return None
        return (x / 1000.0, y / 1000.0)

    def get_card_id(self) -> int | None:
        return self.query_value(MEM_CARD_ID)

    def get_total_work_time(self) -> float | None:
        val = self.query_value(MEM_TOTAL_WORK_TIME)
        return float(val) if val is not None else None

    def get_file_run_time(self, file_num: int = 1) -> float | None:
        addr_val = 0x0390 + file_num
        addr = bytes([(addr_val >> 7) & 0x7F, addr_val & 0x7F])
        val = self.query_value(addr)
        return float(val) if val is not None else None

    def get_doc_count(self) -> int | None:
        return self.query_value(MEM_DOC_COUNT)

    def get_prev_work_time(self) -> float | None:
        """Duration of the last completed job, in seconds."""
        val = self.query_value(MEM_PREV_WORK_TIME)
        return float(val) if val is not None else None

    def start_job(self) -> bool:
        """Start / resume the loaded job."""
        return self._send_command(b"\xD8\x00") is not None

    def stop_job(self) -> bool:
        """Stop the current job."""
        return self._send_command(b"\xD8\x01") is not None


# ---------------------------------------------------------------------------
# Auto-detect Ruida on serial ports
# ---------------------------------------------------------------------------

def find_ruida_serial() -> str | None:
    """Scan serial ports for a Ruida controller. Returns port name or None."""
    import serial.tools.list_ports

    ports = serial.tools.list_ports.comports()
    candidates = []

    for p in ports:
        vid = f"{p.vid:04X}" if p.vid else ""
        desc = (p.description or "").lower()

        # Ruida commonly uses CH340 (1A86:7523) or FTDI (0403:6001)
        if vid in ("1A86", "0403"):
            candidates.append(p.device)
        elif "ch340" in desc or "ch341" in desc or "ruida" in desc:
            candidates.append(p.device)

    # Try each candidate
    for port in candidates:
        conn = RuidaSerialConnection(port=port)
        if conn.connect():
            conn.close()
            return port
        conn.close()

    return None


# ---------------------------------------------------------------------------
# Discovery — find Ruida controllers on the local network
# ---------------------------------------------------------------------------

def discover_ruida(subnet: str = "192.168.1", timeout: float = 1.0) -> list[str]:
    """Scan a /24 subnet for Ruida controllers by sending ENQ to port 50200.

    Returns list of IP addresses that responded with ACK.
    """
    found = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(0.05)

    try:
        sock.bind(("", 0))
    except OSError:
        pass

    enq_pkt = struct.pack(">H", sum(swizzle(b"\xCE")) & 0xFFFF) + swizzle(b"\xCE")

    # Send ENQ to common Ruida addresses first, then sweep
    priority_ips = [f"{subnet}.100", f"{subnet}.1", f"{subnet}.200"]
    all_ips = priority_ips + [
        f"{subnet}.{i}" for i in range(1, 255) if f"{subnet}.{i}" not in priority_ips
    ]

    for ip in all_ips:
        try:
            sock.sendto(enq_pkt, (ip, 50200))
        except Exception:
            continue

    # Collect responses
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            data, (addr, port) = sock.recvfrom(1024)
            if len(data) > 0 and data[0] == 0xC6:
                if addr not in found:
                    found.append(addr)
        except socket.timeout:
            break

    sock.close()
    return found
