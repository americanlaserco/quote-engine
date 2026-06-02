"""Engraving time estimator — area-fill (raster scan) of closed vector shapes.

Cutting and scoring follow a *line*; engraving fills an *area*. The laser
sweeps back and forth in parallel scan-lines, spaced one "line interval"
apart, across the inside of a closed shape. This module computes that fill
from vector geometry (DXF / PDF / AI) and estimates how long it takes.

Added in quote 1.5.

How it works:
  1. Every closed contour of the designated engrave colour is reduced to a
     set of straight edges (curves are linearised by the shared geometry code).
  2. For each horizontal scan-line (one per line interval) the inside spans
     are found with the even-odd rule, so text counters and donut holes are
     left un-engraved correctly.
  3. Per scan-line the head sweeps the fill extent; time is the trapezoidal
     accel/cruise/decel motion for that travel plus a fixed per-line overhead
     for turnaround. A per-job overhead is added once.

CALIBRATION (2026-05-21):
  Fitted against the shop's "engrave test" set — 9 designs measured on the
  Ruida at two speeds (100 and 400 mm/s), 0.1 mm line interval, 18 data
  points total. The fit achieves 4.3% mean error in-sample and 4.5% under
  leave-one-out cross-validation (honest out-of-sample accuracy). Per-span
  and extent/fill-blend refinements were tested and did not improve the
  cross-validated result. Re-run the fit if the machine or process changes.
"""

import math

from laser_time_tool.motion_planner import _extract_segments, _single_move_time

# ---------------------------------------------------------------------------
# Engraving parameters
# ---------------------------------------------------------------------------

# Scan-line spacing (mm). Confirmed correct by the shop. Smaller = finer
# engraving and proportionally more time.
ENGRAVE_INTERVAL_MM = 0.1

# Default head speed during an engraving sweep (mm/s). Set per job in the UI;
# the shop's test data covered 100 and 400 mm/s.
ENGRAVE_SPEED_MM_S = 400.0

# --- the CALIBRATED constants (fit 2026-05-21) -----------------------------

# Extra travel beyond each end of a scan-line. Calibration drove this to 0;
# turnaround is captured by the per-line overhead below instead.
ENGRAVE_OVERSCAN_MM = 0.0

# Effective head acceleration during engraving (mm/s^2). Calibrated — this is
# what makes higher speeds give realistic (sub-linear) time savings.
ENGRAVE_ACCEL_MM_S2 = 1200.0

# Fixed time added per scan-line (s) — direction reversal, settling, the Y
# step, controller processing.
ENGRAVE_LINE_OVERHEAD_S = 0.0

# Fixed time added once per engraving job (s) — startup, head positioning.
ENGRAVE_JOB_OVERHEAD_S = 22.0


class EngravePlannerError(Exception):
    """Raised on malformed engraving input."""
    pass


def _collect_edges(paths):
    """Reduce a list of path dicts to a flat list of closed-contour edges.

    Each edge is ((x1, y1), (x2, y2)). Every contour is forced closed so the
    scan-line fill is well defined.
    """
    edges = []
    for path in paths:
        try:
            segs = _extract_segments(path)
        except Exception:
            continue
        if not segs:
            continue
        for p1, p2 in segs:
            edges.append((p1, p2))
        # Close the contour if the geometry left it open.
        first = segs[0][0]
        last = segs[-1][1]
        if math.hypot(first[0] - last[0], first[1] - last[1]) > 1e-6:
            edges.append((last, first))
    return edges


def _scanline_spans(edges, y):
    """Inside spans [(x0, x1), ...] for the horizontal scan-line at height y.

    Uses the even-odd rule: crossings are sorted and paired, so enclosed
    holes (letter counters, donuts) are correctly left empty.
    """
    xs = []
    for (x1, y1), (x2, y2) in edges:
        ylo, yhi = (y1, y2) if y1 <= y2 else (y2, y1)
        # Half-open interval [ylo, yhi) avoids double-counting shared vertices.
        # Purely horizontal edges (ylo == yhi) never satisfy this and are skipped.
        if ylo <= y < yhi:
            t = (y - y1) / (y2 - y1)
            xs.append(x1 + t * (x2 - x1))
    xs.sort()
    spans = []
    for i in range(0, len(xs) - 1, 2):
        spans.append((xs[i], xs[i + 1]))
    return spans


def estimate_engrave_time(paths,
                          interval=ENGRAVE_INTERVAL_MM,
                          speed=ENGRAVE_SPEED_MM_S,
                          overscan=ENGRAVE_OVERSCAN_MM,
                          accel=ENGRAVE_ACCEL_MM_S2,
                          line_overhead=ENGRAVE_LINE_OVERHEAD_S,
                          job_overhead=ENGRAVE_JOB_OVERHEAD_S):
    """Estimate engraving (area-fill) time for a set of closed vector shapes.

    Args:
        paths: path dicts (the engrave-colour group) to be area-filled.
        interval: scan-line spacing in mm.
        speed: engrave sweep speed in mm/s.
        overscan: extra travel beyond each scan-line end in mm.
        accel: effective head acceleration in mm/s^2.
        line_overhead: fixed seconds added per scan-line.
        job_overhead: fixed seconds added once for the job.

    Returns:
        (seconds, stats) where stats has scanlines, engraved_area_mm2,
        swept_length_mm and fill_height_mm. Returns (0.0, empty) when there
        is nothing to engrave.
    """
    empty = {
        "scanlines": 0,
        "engraved_area_mm2": 0.0,
        "swept_length_mm": 0.0,
        "fill_height_mm": 0.0,
    }

    edges = _collect_edges(paths)
    if not edges or interval <= 0 or speed <= 0:
        return 0.0, empty

    ys = [pt[1] for edge in edges for pt in edge]
    y_min, y_max = min(ys), max(ys)
    if y_max - y_min < 1e-9:
        return 0.0, empty

    total = 0.0
    n_lines = 0
    swept = 0.0        # total head-travel distance (includes overscan)
    on_dist = 0.0      # total laser-on distance (fill area / interval)

    y = y_min + interval / 2.0
    while y < y_max:
        spans = _scanline_spans(edges, y)
        if spans:
            n_lines += 1
            extent = spans[-1][1] - spans[0][0]
            travel = extent + 2.0 * overscan
            total += _single_move_time(travel, speed, accel) + line_overhead
            swept += travel
            on_dist += sum(b - a for a, b in spans)
        y += interval

    if n_lines == 0:
        return 0.0, empty

    total += job_overhead

    stats = {
        "scanlines": n_lines,
        "engraved_area_mm2": round(on_dist * interval, 1),
        "swept_length_mm": round(swept, 1),
        "fill_height_mm": round(y_max - y_min, 1),
    }
    return total, stats
