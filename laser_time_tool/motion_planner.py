"""Offline motion planner for laser cut time estimation.

Uses segment-by-segment trapezoidal motion profiles with junction speed
limits (Grbl/Marlin junction deviation model) to estimate total job time
from DXF geometry alone — no controller needed.
"""

import math

from laser_time_tool.dxf_parser import path_start_point, path_end_point
from laser_time_tool.settings import (
    RAPID_SPEED_MM_S,
    CUTTING_ACCEL_MM_S2,
    RAPID_ACCEL_MM_S2,
    JUNCTION_DEVIATION_MM,
    PATH_OVERHEAD_S,
    JOB_OVERHEAD_S,
)


def estimate_time_offline(
    color_paths: dict[int, list],
    color_speed_map: dict[int, float],
    rapid_speed: float = RAPID_SPEED_MM_S,
    acceleration: float = CUTTING_ACCEL_MM_S2,
    rapid_acceleration: float = RAPID_ACCEL_MM_S2,
    junction_deviation: float = JUNCTION_DEVIATION_MM,
    path_overhead: float = PATH_OVERHEAD_S,
    job_overhead: float = JOB_OVERHEAD_S,
) -> float:
    """
    Estimate job time from geometry and settings alone (no controller needed).

    Uses a segment-by-segment motion planner with junction speed limits
    derived from the Grbl/Marlin junction deviation model.

    Parameters match Ruida controller (from RDWorks screenshots 2026-02-25):
    - rapid_speed: Ruida idle speed (Cut params -> Idle speed)
    - acceleration: Ruida Max Acc * Acc factor 80%
    - rapid_acceleration: Ruida Idle Acc * G0 Acc factor 120%
    - junction_deviation: cornering tolerance calibrated to match Ruida
    - path_overhead: per-path fixed delay for laser on/off,
      controller command processing, and settling time. Calibrated
      against real-world cuts (1/9/100 copy test files) to produce
      estimates ~3% above actual times.
    - Return-to-origin travel is excluded because real-world timers
      measure cut completion, not head return.
    """
    # Normalize coordinates: shift bounding box to start near origin (5,5)
    color_paths = _normalize_paths(color_paths)

    total_seconds = job_overhead
    last_pos = (0.0, 0.0)

    for aci_color, paths in color_paths.items():
        cut_speed = color_speed_map[aci_color]

        for path in paths:
            # Per-path fixed overhead: laser on/off + controller processing
            total_seconds += path_overhead

            # Travel (rapid) to path start — uses idle/G0 acceleration
            start = path_start_point(path)
            travel_dist = _dist(last_pos, start)
            if travel_dist > 0.01:
                total_seconds += _single_move_time(travel_dist, rapid_speed, rapid_acceleration)

            # Cut along path — segment-by-segment with cornering
            segments = _extract_segments(path)
            # Filter out zero/near-zero length segments that create false junctions
            segments = [
                (p1, p2) for p1, p2 in segments
                if _dist(p1, p2) > 0.001
            ]
            if segments:
                total_seconds += _plan_path_time(
                    segments, cut_speed, acceleration, junction_deviation
                )

            last_pos = path_end_point(path)

    return total_seconds


# ---------------------------------------------------------------------------
# Coordinate normalization — shift design to near origin like the laser would
# ---------------------------------------------------------------------------

def _normalize_paths(color_paths: dict[int, list], margin: float = 5.0) -> dict[int, list]:
    """
    Shift all path coordinates so the bounding box starts at (margin, margin).

    DXF files may have arbitrary origins. The real laser software (LightBurn)
    places the design on the bed, typically near the origin. This normalization
    ensures travel distances match reality.
    """
    min_x = float("inf")
    min_y = float("inf")

    for paths in color_paths.values():
        for path in paths:
            for pt in _iter_path_points(path):
                if pt[0] < min_x:
                    min_x = pt[0]
                if pt[1] < min_y:
                    min_y = pt[1]

    if min_x == float("inf"):
        return color_paths

    dx = margin - min_x
    dy = margin - min_y

    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return color_paths

    shifted: dict[int, list] = {}
    for aci, paths in color_paths.items():
        shifted[aci] = [_shift_path(p, dx, dy) for p in paths]
    return shifted


def _iter_path_points(path: dict):
    """Yield all (x, y) points from a path for bounding box calculation."""
    t = path["type"]
    if t == "line":
        yield path["start"]
        yield path["end"]
    elif t == "lwpolyline":
        for seg in path["segments"]:
            yield seg["point"]
    elif t == "polyline":
        yield from path["points"]
    elif t == "circle":
        cx, cy = path["center"]
        r = path["radius"]
        yield (cx - r, cy - r)
        yield (cx + r, cy + r)
    elif t == "arc":
        cx, cy = path["center"]
        r = path["radius"]
        yield (cx - r, cy - r)
        yield (cx + r, cy + r)


def _shift_path(path: dict, dx: float, dy: float) -> dict:
    """Return a copy of the path with all coordinates shifted by (dx, dy)."""
    t = path["type"]

    if t == "line":
        return {
            "type": "line",
            "start": (path["start"][0] + dx, path["start"][1] + dy),
            "end": (path["end"][0] + dx, path["end"][1] + dy),
        }
    elif t == "lwpolyline":
        return {
            "type": "lwpolyline",
            "segments": [
                {"point": (s["point"][0] + dx, s["point"][1] + dy), "bulge": s["bulge"]}
                for s in path["segments"]
            ],
            "closed": path.get("closed", False),
        }
    elif t == "polyline":
        return {
            "type": "polyline",
            "points": [(x + dx, y + dy) for x, y in path["points"]],
            "closed": path.get("closed", False),
        }
    elif t == "circle":
        cx, cy = path["center"]
        return {
            "type": "circle",
            "center": (cx + dx, cy + dy),
            "radius": path["radius"],
        }
    elif t == "arc":
        cx, cy = path["center"]
        return {
            **path,
            "center": (cx + dx, cy + dy),
        }

    return path


# ---------------------------------------------------------------------------
# Segment extraction — convert any path type to a list of (p1, p2) segments
# ---------------------------------------------------------------------------

def _extract_segments(path: dict) -> list[tuple[tuple, tuple]]:
    """Convert a path dict into a list of (start, end) point-pair segments."""
    t = path["type"]

    if t == "line":
        return [(path["start"], path["end"])]

    elif t == "lwpolyline":
        segs = path["segments"]
        result = []
        for i in range(len(segs) - 1):
            p1 = segs[i]["point"]
            p2 = segs[i + 1]["point"]
            bulge = segs[i]["bulge"]
            if abs(bulge) > 1e-6:
                arc_pts = _linearize_bulge(p1, p2, bulge)
                for j in range(len(arc_pts) - 1):
                    result.append((arc_pts[j], arc_pts[j + 1]))
            else:
                result.append((p1, p2))
        return result

    elif t == "polyline":
        pts = path["points"]
        result = []
        for i in range(len(pts) - 1):
            result.append((pts[i], pts[i + 1]))
        if path.get("closed") and len(pts) > 1:
            result.append((pts[-1], pts[0]))
        return result

    elif t == "circle":
        cx, cy = path["center"]
        r = path["radius"]
        n = max(36, int(2 * math.pi * r / 0.5))
        result = []
        for i in range(n):
            a1 = 2 * math.pi * i / n
            a2 = 2 * math.pi * (i + 1) / n
            result.append((
                (cx + r * math.cos(a1), cy + r * math.sin(a1)),
                (cx + r * math.cos(a2), cy + r * math.sin(a2)),
            ))
        return result

    elif t == "arc":
        cx, cy = path["center"]
        r = path["radius"]
        start_deg = path["start_deg"]
        end_deg = path["end_deg"]
        sweep = end_deg - start_deg
        if sweep < 0:
            sweep += 360
        n = max(8, int(sweep / 5))
        result = []
        for i in range(n):
            a1 = math.radians(start_deg + sweep * i / n)
            a2 = math.radians(start_deg + sweep * (i + 1) / n)
            result.append((
                (cx + r * math.cos(a1), cy + r * math.sin(a1)),
                (cx + r * math.cos(a2), cy + r * math.sin(a2)),
            ))
        return result

    return []


def _linearize_bulge(
    p1: tuple, p2: tuple, bulge: float, tol: float = 0.1
) -> list[tuple]:
    """Linearize a bulge arc into a polyline of short chords."""
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


# ---------------------------------------------------------------------------
# Segment-by-segment motion planner with junction speed limits
# ---------------------------------------------------------------------------

def _plan_path_time(
    segments: list[tuple[tuple, tuple]],
    max_speed: float,
    acceleration: float,
    junction_deviation: float,
) -> float:
    """
    Compute total time for a sequence of linear segments using the
    junction deviation model (same concept used by Grbl, Marlin, RRF).
    """
    n = len(segments)
    if n == 0:
        return 0.0

    # --- lengths and unit directions ---
    lengths = []
    dirs = []
    for (sx, sy), (ex, ey) in segments:
        dx = ex - sx
        dy = ey - sy
        length = math.sqrt(dx * dx + dy * dy)
        lengths.append(length)
        if length > 1e-9:
            dirs.append((dx / length, dy / length))
        else:
            dirs.append((0.0, 0.0))

    # --- junction speeds (n+1 junctions: start, between segments, end) ---
    jv = [0.0] * (n + 1)
    jv[n] = 0.0

    for i in range(1, n):
        d1 = dirs[i - 1]
        d2 = dirs[i]
        cos_theta = d1[0] * d2[0] + d1[1] * d2[1]
        cos_theta = max(-1.0, min(1.0, cos_theta))

        if cos_theta >= 0.9999:
            jv[i] = max_speed
        elif cos_theta <= -0.9999:
            jv[i] = 0.0
        else:
            sin_theta_d2 = math.sqrt(0.5 * (1.0 + cos_theta))
            if sin_theta_d2 > 0.9999:
                jv[i] = max_speed
            elif sin_theta_d2 < 1e-6:
                jv[i] = 0.0
            else:
                jv[i] = math.sqrt(
                    acceleration * junction_deviation
                    * sin_theta_d2 / (1.0 - sin_theta_d2)
                )
                jv[i] = min(jv[i], max_speed)

    # --- forward pass: can't exceed what we can accelerate to ---
    for i in range(1, n + 1):
        if lengths[i - 1] < 1e-9:
            jv[i] = min(jv[i], jv[i - 1])
            continue
        v_reachable = math.sqrt(jv[i - 1] ** 2 + 2.0 * acceleration * lengths[i - 1])
        jv[i] = min(jv[i], v_reachable, max_speed)

    # --- backward pass: can't exceed what we can decelerate from ---
    for i in range(n - 1, -1, -1):
        if lengths[i] < 1e-9:
            jv[i] = min(jv[i], jv[i + 1])
            continue
        v_reachable = math.sqrt(jv[i + 1] ** 2 + 2.0 * acceleration * lengths[i])
        jv[i] = min(jv[i], v_reachable, max_speed)

    # --- sum per-segment time ---
    total = 0.0
    for i in range(n):
        if lengths[i] < 1e-9:
            continue
        total += _segment_time(lengths[i], jv[i], jv[i + 1], max_speed, acceleration)

    return total


def _segment_time(
    dist: float,
    v_entry: float,
    v_exit: float,
    v_max: float,
    accel: float,
) -> float:
    """
    Time for one segment: accelerate from v_entry, cruise at v_peak, decelerate to v_exit.
    """
    if dist < 1e-9:
        return 0.0

    v_peak_sq = (2.0 * accel * dist + v_entry ** 2 + v_exit ** 2) / 2.0
    v_peak_sq = max(v_peak_sq, 0.0)
    v_peak = min(math.sqrt(v_peak_sq), v_max)
    v_peak = max(v_peak, v_entry, v_exit)

    d_accel = (v_peak ** 2 - v_entry ** 2) / (2.0 * accel) if v_peak > v_entry else 0.0
    d_decel = (v_peak ** 2 - v_exit ** 2) / (2.0 * accel) if v_peak > v_exit else 0.0
    d_accel = max(d_accel, 0.0)
    d_decel = max(d_decel, 0.0)
    d_cruise = max(dist - d_accel - d_decel, 0.0)

    t_accel = (v_peak - v_entry) / accel if v_peak > v_entry else 0.0
    t_decel = (v_peak - v_exit) / accel if v_peak > v_exit else 0.0
    t_cruise = d_cruise / v_peak if v_peak > 1e-9 else 0.0

    return t_accel + t_cruise + t_decel


def _single_move_time(distance: float, max_speed: float, acceleration: float) -> float:
    """Time for a point-to-point rapid move (start and end at rest)."""
    return _segment_time(distance, 0.0, 0.0, max_speed, acceleration)


def _dist(p1: tuple, p2: tuple) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)
