"""Parse DXF files and extract geometric paths grouped by ACI color."""

import logging
import math
from pathlib import Path

import ezdxf
from ezdxf.entities import (
    Arc,
    Circle,
    Ellipse,
    Insert,
    Line,
    LWPolyline,
    Polyline,
    Spline,
)

from laser_time_tool.settings import resolve_entity_color

logger = logging.getLogger(__name__)


class DXFParseError(Exception):
    """Raised when a DXF file cannot be read or is structurally invalid."""
    pass


def parse_dxf(filepath: str, color_speed_map: dict[int, float]) -> dict[int, list]:
    """
    Parse a DXF file and return paths grouped by ACI color number.

    Only colors present in color_speed_map are included in the output.
    Returns a dict of {aci_color: [list of path dicts]}.

    Raises DXFParseError for file-level errors (missing, corrupt, not a DXF).
    Skips individual malformed entities with a warning.
    """
    try:
        doc = ezdxf.readfile(filepath)
    except FileNotFoundError:
        raise DXFParseError(f"DXF file not found: {filepath}")
    except ezdxf.DXFStructureError as e:
        raise DXFParseError(f"Invalid DXF structure in {filepath}: {e}")
    except ezdxf.DXFError as e:
        raise DXFParseError(f"Cannot read DXF file {filepath}: {e}")
    except (OSError, IOError) as e:
        raise DXFParseError(f"Cannot open {filepath}: {e}")

    msp = doc.modelspace()

    # Determine unit scale factor (convert to mm)
    scale = _get_unit_scale(doc)

    color_paths: dict[int, list] = {}

    # Collect all entities, exploding block references
    entities = list(_iter_entities(msp, doc))

    for entity in entities:
        try:
            aci = resolve_entity_color(entity, doc)
        except Exception:
            logger.warning("Could not resolve color for entity %s, skipping", entity.dxftype())
            continue

        if aci == 0:
            continue

        if aci not in color_speed_map:
            continue

        try:
            paths = _entity_to_paths(entity, scale)
        except Exception:
            logger.warning("Failed to convert %s entity on layer '%s', skipping",
                          entity.dxftype(), entity.dxf.get("layer", "0"))
            continue

        if paths:
            color_paths.setdefault(aci, []).extend(paths)

    return color_paths


def scan_dxf_colors(dxf_files: list[Path], color_speed_map: dict[int, float]) -> tuple[dict[int, int], list[str]]:
    """
    Scan multiple DXF files and count entities by color.

    Returns:
        (color_counts, warnings) where color_counts = {aci: entity_count}
    """
    from laser_time_tool.settings import ACI_TO_NAME

    color_counts: dict[int, int] = {}
    unrecognized: dict[int, int] = {}
    warnings: list[str] = []

    for dxf_path in dxf_files:
        try:
            doc = ezdxf.readfile(str(dxf_path))
        except (ezdxf.DXFError, OSError) as e:
            warnings.append(f"WARNING: Cannot read {dxf_path.name}: {e}")
            continue

        msp = doc.modelspace()
        entities = list(_iter_entities(msp, doc))

        for entity in entities:
            try:
                aci = resolve_entity_color(entity, doc)
            except Exception:
                continue
            if aci in color_speed_map:
                color_counts[aci] = color_counts.get(aci, 0) + 1
            elif aci != 0:
                unrecognized[aci] = unrecognized.get(aci, 0) + 1

    for aci, count in unrecognized.items():
        name = ACI_TO_NAME.get(aci, f"ACI {aci}")
        warnings.append(
            f"WARNING: {count} entities with unrecognized color ({name}, ACI {aci}) — will be SKIPPED"
        )

    return color_counts, warnings


def _get_unit_scale(doc) -> float:
    """Get scale factor to convert DXF units to millimeters."""
    try:
        units = doc.header.get("$INSUNITS", 0)
    except Exception:
        units = 0

    # DXF unit codes: 0=unspecified, 1=inches, 2=feet, 3=miles, 4=mm, 5=cm, 6=m
    scale_map = {
        0: 1.0,   # Unspecified — assume mm
        1: 25.4,  # Inches to mm
        2: 304.8, # Feet to mm
        4: 1.0,   # Already mm
        5: 10.0,  # cm to mm
        6: 1000.0, # m to mm
    }
    return scale_map.get(units, 1.0)


def _iter_entities(layout, doc, depth=0):
    """Iterate entities, exploding INSERT (block) references."""
    if depth > 10:  # Prevent infinite recursion
        return

    for entity in layout:
        if isinstance(entity, Insert):
            try:
                for sub in entity.virtual_entities():
                    yield sub
            except Exception:
                pass
        else:
            yield entity


def _entity_to_paths(entity, scale: float) -> list[dict]:
    """Convert a DXF entity to path dict(s).

    Returns an empty list for unsupported entity types.
    Raises on truly malformed geometry (caught by caller).
    """
    if isinstance(entity, Line):
        path = _parse_line(entity, scale)
        if _is_degenerate_line(path):
            return []
        return [path]
    elif isinstance(entity, LWPolyline):
        path = _parse_lwpolyline(entity, scale)
        if not path["segments"]:
            return []
        return [path]
    elif isinstance(entity, Polyline):
        path = _parse_polyline(entity, scale)
        if not path["points"]:
            return []
        return [path]
    elif isinstance(entity, Circle):
        path = _parse_circle(entity, scale)
        if path["radius"] <= 0:
            return []
        return [path]
    elif isinstance(entity, Arc):
        path = _parse_arc(entity, scale)
        if path["radius"] <= 0:
            return []
        return [path]
    elif isinstance(entity, Spline):
        path = _parse_spline(entity, scale)
        if len(path["points"]) < 2:
            return []
        return [path]
    elif isinstance(entity, Ellipse):
        path = _parse_ellipse(entity, scale)
        if len(path["points"]) < 2:
            return []
        return [path]
    return []


def _is_degenerate_line(path: dict) -> bool:
    """Check if a line path has zero length."""
    sx, sy = path["start"]
    ex, ey = path["end"]
    return abs(sx - ex) < 1e-9 and abs(sy - ey) < 1e-9


def _parse_line(entity: Line, scale: float) -> dict:
    start = entity.dxf.start
    end = entity.dxf.end
    return {
        "type": "line",
        "start": (start.x * scale, start.y * scale),
        "end": (end.x * scale, end.y * scale),
    }


def _parse_lwpolyline(entity: LWPolyline, scale: float) -> dict:
    """Parse LWPOLYLINE including bulge (arc) segments."""
    # Get vertices as (x, y, start_width, end_width, bulge)
    points_data = list(entity.get_points(format="xyseb"))
    closed = entity.closed

    segments = []
    for i in range(len(points_data)):
        x, y, _sw, _ew, bulge = points_data[i]
        p1 = (x * scale, y * scale)

        # Determine next point
        if i + 1 < len(points_data):
            nx, ny = points_data[i + 1][0] * scale, points_data[i + 1][1] * scale
        elif closed and len(points_data) > 1:
            nx, ny = points_data[0][0] * scale, points_data[0][1] * scale
        else:
            # Last point, no next segment
            segments.append({"point": p1, "bulge": 0})
            break

        segments.append({"point": p1, "bulge": bulge})

    # Add closing point if closed
    if closed and len(points_data) > 1:
        segments.append({"point": (points_data[0][0] * scale, points_data[0][1] * scale), "bulge": 0})

    return {
        "type": "lwpolyline",
        "segments": segments,
        "closed": closed,
    }


def _parse_polyline(entity: Polyline, scale: float) -> dict:
    """Parse old-style POLYLINE entity."""
    points = [(v.dxf.location.x * scale, v.dxf.location.y * scale) for v in entity.vertices]
    closed = entity.is_closed
    return {
        "type": "polyline",
        "points": points,
        "closed": closed,
    }


def _parse_circle(entity: Circle, scale: float) -> dict:
    center = entity.dxf.center
    return {
        "type": "circle",
        "center": (center.x * scale, center.y * scale),
        "radius": entity.dxf.radius * scale,
    }


def _parse_arc(entity: Arc, scale: float) -> dict:
    center = entity.dxf.center
    return {
        "type": "arc",
        "center": (center.x * scale, center.y * scale),
        "radius": entity.dxf.radius * scale,
        "start_deg": entity.dxf.start_angle,
        "end_deg": entity.dxf.end_angle,
    }


def _parse_spline(entity: Spline, scale: float) -> dict:
    """Flatten spline to line segments."""
    try:
        points = [(p.x * scale, p.y * scale) for p in entity.flattening(0.05)]
    except Exception:
        # Fallback: use control points
        points = [(p.x * scale, p.y * scale) for p in entity.control_points]
    return {
        "type": "polyline",
        "points": points,
        "closed": entity.closed,
    }


def _parse_ellipse(entity: Ellipse, scale: float) -> dict:
    """Flatten ellipse to line segments."""
    try:
        points = [(p.x * scale, p.y * scale) for p in entity.flattening(0.05)]
    except Exception:
        # Approximate as a polyline from vertices
        points = [(p.x * scale, p.y * scale) for p in entity.vertices(entity.dxf.get("count", 64))]
    return {
        "type": "polyline",
        "points": points,
        "closed": True,
    }


def path_start_point(path: dict) -> tuple[float, float]:
    """Get the starting point of a path."""
    t = path["type"]
    if t == "line":
        return path["start"]
    elif t == "lwpolyline":
        if path["segments"]:
            return path["segments"][0]["point"]
        return (0.0, 0.0)
    elif t == "polyline":
        if path["points"]:
            return path["points"][0]
        return (0.0, 0.0)
    elif t == "circle":
        cx, cy = path["center"]
        return (cx, cy + path["radius"])  # Top of circle
    elif t == "arc":
        cx, cy = path["center"]
        r = path["radius"]
        a = math.radians(path["start_deg"])
        return (cx + r * math.cos(a), cy + r * math.sin(a))
    return (0.0, 0.0)


def path_end_point(path: dict) -> tuple[float, float]:
    """Get the ending point of a path."""
    t = path["type"]
    if t == "line":
        return path["end"]
    elif t == "lwpolyline":
        if path["segments"]:
            return path["segments"][-1]["point"]
        return (0.0, 0.0)
    elif t == "polyline":
        if path["points"]:
            return path["points"][-1]
        return (0.0, 0.0)
    elif t == "circle":
        cx, cy = path["center"]
        return (cx, cy + path["radius"])  # Back to top
    elif t == "arc":
        cx, cy = path["center"]
        r = path["radius"]
        a = math.radians(path["end_deg"])
        return (cx + r * math.cos(a), cy + r * math.sin(a))
    return (0.0, 0.0)


def path_length(path: dict) -> float:
    """Calculate the total travel distance of a path in mm."""
    t = path["type"]

    if t == "line":
        return _dist(path["start"], path["end"])

    elif t == "lwpolyline":
        total = 0.0
        segs = path["segments"]
        for i in range(len(segs) - 1):
            p1 = segs[i]["point"]
            p2 = segs[i + 1]["point"]
            bulge = segs[i]["bulge"]
            if abs(bulge) > 1e-6:
                total += _bulge_arc_length(p1, p2, bulge)
            else:
                total += _dist(p1, p2)
        return total

    elif t == "polyline":
        total = 0.0
        pts = path["points"]
        for i in range(len(pts) - 1):
            total += _dist(pts[i], pts[i + 1])
        if path.get("closed") and len(pts) > 1:
            total += _dist(pts[-1], pts[0])
        return total

    elif t == "circle":
        return 2 * math.pi * path["radius"]

    elif t == "arc":
        r = path["radius"]
        start = path["start_deg"]
        end = path["end_deg"]
        angle = end - start
        if angle < 0:
            angle += 360
        return r * math.radians(angle)

    return 0.0


def _dist(p1: tuple, p2: tuple) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def _bulge_arc_length(p1: tuple, p2: tuple, bulge: float) -> float:
    """Calculate arc length from bulge value between two points."""
    chord = _dist(p1, p2)
    if chord < 1e-9:
        return 0.0
    # included_angle = 4 * atan(|bulge|)
    included_angle = 4 * math.atan(abs(bulge))
    if abs(included_angle) < 1e-9:
        return chord
    radius = chord / (2 * math.sin(included_angle / 2))
    return abs(radius * included_angle)
