"""Build Ruida .rd binary job files from parsed DXF paths.

Generates the proprietary binary format used by Ruida laser controllers
(RDC6442, RDC6445, etc.). The output is scrambled and ready for UDP upload.

References:
  - meerk40t/ruida/rdjob.py (MIT license)
  - jnweiger/ruida-laser/src/ruida.py
  - kkaempf/ruida/doc/commands.md
"""

import math

from laser_time_tool.ruida_connection import encode32, encode_coord, encode_speed, encode_power, encode_color, swizzle

# ---------------------------------------------------------------------------
# ACI color to RGB mapping (for Ruida layer colors)
# ---------------------------------------------------------------------------

ACI_TO_RGB = {
    1: (255, 0, 0),       # Red
    2: (255, 255, 0),     # Yellow
    3: (0, 255, 0),       # Green
    4: (0, 255, 255),     # Cyan
    5: (0, 0, 255),       # Blue
    6: (255, 0, 255),     # Magenta
    7: (0, 0, 0),         # Black (White/Black)
}

# Default layer colors used by RDWorks for layers 0-6
LAYER_COLORS_RGB = [
    (0, 0, 0),        # Layer 0: Black
    (255, 0, 0),      # Layer 1: Red
    (0, 255, 0),      # Layer 2: Green
    (255, 255, 0),    # Layer 3: Yellow
    (0, 0, 255),      # Layer 4: Blue
    (255, 0, 255),    # Layer 5: Magenta
    (0, 255, 255),    # Layer 6: Cyan
]


class RdFileBuilder:
    """Builds Ruida .rd binary job files from parsed DXF paths."""

    def __init__(self):
        self.commands: list[bytes] = []

    def _cmd(self, *data: int | bytes) -> None:
        """Append a command (sequence of bytes) to the buffer."""
        buf = bytearray()
        for d in data:
            if isinstance(d, (bytes, bytearray)):
                buf.extend(d)
            else:
                buf.append(d & 0xFF)
        self.commands.append(bytes(buf))

    # -------------------------------------------------------------------
    # Header commands
    # -------------------------------------------------------------------

    def add_header(
        self,
        x_min_mm: float,
        y_min_mm: float,
        x_max_mm: float,
        y_max_mm: float,
    ) -> None:
        """Add file header with bounding box and standard setup."""
        # File type: block mode
        self._cmd(0xF1, 0x02, 0x00)

        # Reference point (origin)
        self._cmd(0xE7, 0x06, encode_coord(0), encode_coord(0))

        # Bounding box — top-left and bottom-right
        self._cmd(0xE7, 0x03, encode_coord(x_min_mm), encode_coord(y_min_mm))
        self._cmd(0xE7, 0x07, encode_coord(x_max_mm), encode_coord(y_max_mm))

        # Extended min/max points
        self._cmd(0xE7, 0x50, encode_coord(x_min_mm), encode_coord(y_min_mm))
        self._cmd(0xE7, 0x51, encode_coord(x_max_mm), encode_coord(y_max_mm))

        # Array repeat count = 1
        self._cmd(0xE7, 0x04, encode32(1))

    # -------------------------------------------------------------------
    # Layer setup
    # -------------------------------------------------------------------

    def add_layer(
        self,
        layer_num: int,
        speed_mm_s: float,
        power_min_percent: float = 10.0,
        power_max_percent: float = 50.0,
        color_rgb: tuple[int, int, int] = (0, 0, 0),
        repeat_count: int = 1,
    ) -> None:
        """Add layer configuration (speed, power, color, work mode)."""
        # Layer number (0-based)
        self._cmd(0xCA, 0x02, layer_num & 0x7F)

        # Part speed
        self._cmd(0xC9, 0x04, encode_speed(speed_mm_s))

        # Part power — laser 1 min/max
        self._cmd(0xC6, 0x31, encode_power(power_min_percent))
        self._cmd(0xC6, 0x32, encode_power(power_max_percent))

        # Part power — laser 2 min/max (set same as laser 1)
        self._cmd(0xC6, 0x35, encode_power(power_min_percent))
        self._cmd(0xC6, 0x36, encode_power(power_max_percent))

        # Layer repeat count
        self._cmd(0xCA, 0x06, encode32(repeat_count))

        # Layer color
        r, g, b = color_rgb
        self._cmd(0xCA, 0x05, encode_color(r, g, b))

        # Work mode = cutting (1)
        self._cmd(0xCA, 0x01, 0x01)

    # -------------------------------------------------------------------
    # Movement commands
    # -------------------------------------------------------------------

    def add_move(self, x_mm: float, y_mm: float) -> None:
        """Add rapid move (laser OFF) to absolute position."""
        self._cmd(0x88, encode_coord(x_mm), encode_coord(y_mm))

    def add_cut(self, x_mm: float, y_mm: float) -> None:
        """Add cut move (laser ON) to absolute position."""
        self._cmd(0xA8, encode_coord(x_mm), encode_coord(y_mm))

    # -------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------

    def add_layer_end(self) -> None:
        """Mark end of current layer's data."""
        self._cmd(0xCA, 0x01, 0x00)

    def add_footer(self) -> None:
        """Add file footer (block end, array end, checksum, EOF)."""
        # Block end
        self._cmd(0xE7, 0x00)

        # Array end
        self._cmd(0xEB)

        # File checksum: sum of all command bytes + 0xD7 (the EOF byte)
        total = sum(sum(cmd) for cmd in self.commands) + 0xD7
        self._cmd(0xE5, 0x05, encode32(total))

        # EOF marker
        self._cmd(0xD7)

    # -------------------------------------------------------------------
    # Build the complete file
    # -------------------------------------------------------------------

    def build(self) -> bytes:
        """Build the complete .rd file as scrambled bytes, ready for upload."""
        raw = b"".join(self.commands)
        return swizzle(raw)

    def build_raw(self) -> bytes:
        """Build the complete .rd file as unscrambled bytes (for debugging)."""
        return b"".join(self.commands)


# ---------------------------------------------------------------------------
# High-level: convert parsed DXF paths to .rd file
# ---------------------------------------------------------------------------

def build_rd_from_paths(
    color_paths: dict[int, list],
    color_speed_map: dict[int, float],
    power_min: float = 10.0,
    power_max: float = 50.0,
) -> bytes:
    """
    Convert parsed DXF paths to a scrambled .rd binary file.

    color_paths: {aci_color: [list of path dicts]} from dxf_parser
    color_speed_map: {aci_color: speed_mm_s} from job.txt

    Returns scrambled bytes ready for UDP upload to Ruida controller.
    """
    builder = RdFileBuilder()

    # Calculate bounding box across all paths
    x_min, y_min, x_max, y_max = _compute_bounding_box(color_paths)

    # Add a small margin
    margin = 1.0
    builder.add_header(
        max(0, x_min - margin),
        max(0, y_min - margin),
        x_max + margin,
        y_max + margin,
    )

    # Process each color as a separate layer
    layer_num = 0
    for aci_color, paths in color_paths.items():
        speed = color_speed_map.get(aci_color, 10.0)
        rgb = ACI_TO_RGB.get(aci_color, (0, 0, 0))

        builder.add_layer(
            layer_num=layer_num,
            speed_mm_s=speed,
            power_min_percent=power_min,
            power_max_percent=power_max,
            color_rgb=rgb,
        )

        # Convert each path to move/cut commands
        for path in paths:
            _add_path_commands(builder, path)

        builder.add_layer_end()
        layer_num += 1

    builder.add_footer()
    return builder.build()


def _compute_bounding_box(
    color_paths: dict[int, list],
) -> tuple[float, float, float, float]:
    """Compute overall bounding box of all paths in mm."""
    x_min = float("inf")
    y_min = float("inf")
    x_max = float("-inf")
    y_max = float("-inf")

    for paths in color_paths.values():
        for path in paths:
            points = _get_all_points(path)
            for x, y in points:
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

    if x_min == float("inf"):
        return (0, 0, 100, 100)

    return (x_min, y_min, x_max, y_max)


def _get_all_points(path: dict) -> list[tuple[float, float]]:
    """Extract all coordinate points from a path for bounding box."""
    t = path["type"]
    if t == "line":
        return [path["start"], path["end"]]
    elif t == "lwpolyline":
        return [s["point"] for s in path["segments"]]
    elif t == "polyline":
        return path["points"]
    elif t == "circle":
        cx, cy = path["center"]
        r = path["radius"]
        return [(cx - r, cy - r), (cx + r, cy + r)]
    elif t == "arc":
        cx, cy = path["center"]
        r = path["radius"]
        return [(cx - r, cy - r), (cx + r, cy + r)]
    return []


def _add_path_commands(builder: RdFileBuilder, path: dict) -> None:
    """Convert a single path to move/cut commands on the builder."""
    t = path["type"]

    if t == "line":
        sx, sy = path["start"]
        ex, ey = path["end"]
        builder.add_move(sx, sy)
        builder.add_cut(ex, ey)

    elif t == "lwpolyline":
        _add_lwpolyline_commands(builder, path)

    elif t == "polyline":
        _add_polyline_commands(builder, path)

    elif t == "circle":
        _add_circle_commands(builder, path)

    elif t == "arc":
        _add_arc_commands(builder, path)


def _add_lwpolyline_commands(builder: RdFileBuilder, path: dict) -> None:
    """Convert LWPOLYLINE (with bulge arcs) to move/cut commands."""
    segments = path["segments"]
    if not segments:
        return

    # Move to start
    sx, sy = segments[0]["point"]
    builder.add_move(sx, sy)

    for i in range(len(segments) - 1):
        p1 = segments[i]["point"]
        p2 = segments[i + 1]["point"]
        bulge = segments[i]["bulge"]

        if abs(bulge) > 1e-6:
            # Linearize arc into small cuts
            arc_pts = _linearize_bulge_arc(p1, p2, bulge)
            for pt in arc_pts[1:]:  # Skip first point (already there)
                builder.add_cut(pt[0], pt[1])
        else:
            builder.add_cut(p2[0], p2[1])


def _add_polyline_commands(builder: RdFileBuilder, path: dict) -> None:
    """Convert polyline to move/cut commands."""
    points = path["points"]
    if not points:
        return

    builder.add_move(points[0][0], points[0][1])
    for x, y in points[1:]:
        builder.add_cut(x, y)

    if path.get("closed") and len(points) > 1:
        builder.add_cut(points[0][0], points[0][1])


def _add_circle_commands(builder: RdFileBuilder, path: dict) -> None:
    """Convert circle to linearized cut commands."""
    cx, cy = path["center"]
    r = path["radius"]

    # Linearize into segments (more segments for larger circles)
    n = max(36, int(2 * math.pi * r / 0.5))

    # Start at top of circle
    start_x = cx + r
    start_y = cy
    builder.add_move(start_x, start_y)

    for i in range(1, n + 1):
        angle = 2 * math.pi * i / n
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        builder.add_cut(x, y)


def _add_arc_commands(builder: RdFileBuilder, path: dict) -> None:
    """Convert arc to linearized cut commands."""
    cx, cy = path["center"]
    r = path["radius"]
    start_deg = path["start_deg"]
    end_deg = path["end_deg"]

    sweep = end_deg - start_deg
    if sweep < 0:
        sweep += 360

    n = max(8, int(sweep / 5))

    # Move to arc start
    start_rad = math.radians(start_deg)
    builder.add_move(cx + r * math.cos(start_rad), cy + r * math.sin(start_rad))

    for i in range(1, n + 1):
        angle = math.radians(start_deg + sweep * i / n)
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        builder.add_cut(x, y)


def _linearize_bulge_arc(
    p1: tuple, p2: tuple, bulge: float, tol: float = 0.1
) -> list[tuple[float, float]]:
    """Linearize a bulge arc into a list of points."""
    from laser_time_tool.gcode_generator import bulge_to_arc

    cx, cy, radius, clockwise = bulge_to_arc(p1, p2, bulge)
    if radius < 1e-9:
        return [p1, p2]

    included = 4 * math.atan(abs(bulge))
    arc_len = radius * included
    n = max(4, int(arc_len / tol))

    a1 = math.atan2(p1[1] - cy, p1[0] - cx)
    a2 = math.atan2(p2[1] - cy, p2[0] - cx)

    if clockwise:
        if a1 < a2:
            a1 += 2 * math.pi
        angles = [a1 + (a2 - a1) * i / n for i in range(n + 1)]
    else:
        if a2 < a1:
            a2 += 2 * math.pi
        angles = [a1 + (a2 - a1) * i / n for i in range(n + 1)]

    return [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]
