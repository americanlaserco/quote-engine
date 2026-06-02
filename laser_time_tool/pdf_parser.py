"""Parse PDF and Adobe Illustrator (.ai) files into geometric paths.

Native vector reader — requires no external software. Uses PyMuPDF (fitz) to
read the vector drawing operators directly out of a PDF/AI file and converts
them into the exact path-dict format produced by ``dxf_parser`` so the rest of
the engine (path length, motion planner, calibration) consumes them unchanged.

Added in quote 1.4 to let the quoting engine accept .pdf and .ai uploads
alongside .dxf. Modern .ai files are PDF-compatible internally and are read
the same way.
"""

import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

# PDF user space is 1/72 inch per unit. Multiply a coordinate by this to get mm.
PT_TO_MM = 25.4 / 72.0

# Max distance (mm) a flattened chord may deviate from the true bezier curve.
# Kept in the same ballpark as the DXF spline/bulge flattening so the motion
# planner sees a comparable segment density (keeps the calibration consistent).
FLATTEN_TOL_MM = 0.05

# Two points within this distance (mm) are treated as the same point when
# chaining drawing items into a single connected contour.
JOIN_TOL_MM = 0.05

# Recursion cap for adaptive bezier flattening (2^16 segments worst case).
_MAX_FLATTEN_DEPTH = 16


class PDFParseError(Exception):
    """Raised when a PDF/AI file cannot be read or has no usable geometry."""
    pass


# ---------------------------------------------------------------------------
# Colour mapping — snap RGB strokes to AutoCAD Color Index (ACI) numbers
# ---------------------------------------------------------------------------

# Standard ACI colours as RGB (0-1). Black maps to ACI 7; white also resolves
# to ACI 7, matching COLOR_NAME_TO_ACI in settings.py.
_ACI_RGB = {
    1: (1.0, 0.0, 0.0),   # red
    2: (1.0, 1.0, 0.0),   # yellow
    3: (0.0, 1.0, 0.0),   # green
    4: (0.0, 1.0, 1.0),   # cyan
    5: (0.0, 0.0, 1.0),   # blue
    6: (1.0, 0.0, 1.0),   # magenta
    7: (0.0, 0.0, 0.0),   # black -> ACI 7
}


def rgb_to_aci(rgb) -> int:
    """Snap an RGB triple (0-1 floats) to the nearest standard ACI colour.

    Pure black and pure white both resolve to ACI 7, consistent with the DXF
    colour scheme used elsewhere in the engine. ``None`` resolves to ACI 7.
    """
    if rgb is None:
        return 7
    try:
        r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    except (TypeError, ValueError, IndexError):
        return 7

    best_aci = 7
    best_dist = float("inf")
    for aci, (cr, cg, cb) in _ACI_RGB.items():
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_dist:
            best_dist = d
            best_aci = aci

    # White (1,1,1) is also ACI 7.
    d_white = (r - 1.0) ** 2 + (g - 1.0) ** 2 + (b - 1.0) ** 2
    if d_white < best_dist:
        best_aci = 7
    return best_aci


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _close(a, b) -> bool:
    """True if two points are within the join tolerance."""
    return _dist(a, b) <= JOIN_TOL_MM


def _mid(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _point_line_dist(p, a, b) -> float:
    """Perpendicular distance from point p to the line through a and b."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    chord = math.hypot(dx, dy)
    if chord < 1e-12:
        return _dist(p, a)
    return abs((p[0] - ax) * dy - (p[1] - ay) * dx) / chord


def _flatten_cubic(p0, p1, p2, p3, tol):
    """Adaptively flatten a cubic bezier into points.

    Returns the points AFTER p0 (p0 itself is excluded so the result can be
    appended directly onto an existing contour).
    """
    out = []

    def recurse(a, b, c, d, depth):
        if depth >= _MAX_FLATTEN_DEPTH:
            out.append(d)
            return
        # Flat enough when both control points lie close to the a-d chord.
        if (_point_line_dist(b, a, d) <= tol
                and _point_line_dist(c, a, d) <= tol):
            out.append(d)
            return
        # de Casteljau subdivision at t = 0.5
        ab = _mid(a, b)
        bc = _mid(b, c)
        cd = _mid(c, d)
        abc = _mid(ab, bc)
        bcd = _mid(bc, cd)
        m = _mid(abc, bcd)
        recurse(a, ab, abc, m, depth + 1)
        recurse(m, bcd, cd, d, depth + 1)

    recurse(p0, p1, p2, p3, 0)
    return out


def _pt(point, scale):
    """Convert a fitz.Point (or (x, y)) to a scaled (x, y) tuple in mm."""
    try:
        return (point.x * scale, point.y * scale)
    except AttributeError:
        return (point[0] * scale, point[1] * scale)


# ---------------------------------------------------------------------------
# Drawing-item -> contour conversion
# ---------------------------------------------------------------------------

def _items_to_contours(items, scale, closed_hint):
    """Turn a drawing's item list into a list of (points, closed) contours.

    Consecutive line/curve items that join end-to-start are merged into a
    single polyline contour. Rectangles and quads become their own closed
    contours. All coordinates are returned in millimetres.
    """
    contours = []
    current = []

    def flush():
        nonlocal current
        if len(current) >= 2:
            if _close(current[0], current[-1]):
                # Endpoints already coincide -> the contour is physically
                # closed; no extra closing segment needed.
                contours.append((current, False))
            else:
                contours.append((current, closed_hint))
        current = []

    for it in items:
        op = it[0]

        if op == "l":  # straight line
            p1 = _pt(it[1], scale)
            p2 = _pt(it[2], scale)
            if current and _close(current[-1], p1):
                current.append(p2)
            else:
                flush()
                current = [p1, p2]

        elif op == "c":  # cubic bezier
            p0 = _pt(it[1], scale)
            c1 = _pt(it[2], scale)
            c2 = _pt(it[3], scale)
            p3 = _pt(it[4], scale)
            if current and _close(current[-1], p0):
                current.extend(_flatten_cubic(current[-1], c1, c2, p3,
                                              FLATTEN_TOL_MM))
            else:
                flush()
                current = [p0]
                current.extend(_flatten_cubic(p0, c1, c2, p3, FLATTEN_TOL_MM))

        elif op == "re":  # rectangle
            flush()
            rect = it[1]
            x0, y0 = rect.x0 * scale, rect.y0 * scale
            x1, y1 = rect.x1 * scale, rect.y1 * scale
            contours.append(([
                (x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0),
            ], False))

        elif op == "qu":  # quad
            flush()
            q = it[1]
            contours.append(([
                _pt(q.ul, scale), _pt(q.ur, scale), _pt(q.lr, scale),
                _pt(q.ll, scale), _pt(q.ul, scale),
            ], False))

        # Unknown operators are ignored.

    flush()
    return contours


def _extract_page_paths(page, scale):
    """Yield (aci, path_dict) for every cuttable contour on a page."""
    try:
        drawings = page.get_drawings()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not read drawings from page: %s", e)
        return

    for d in drawings:
        dtype = d.get("type", "s")
        if dtype == "clip":
            continue  # clipping paths are masks, not cut geometry

        stroke = d.get("color")
        fill = d.get("fill")
        if dtype == "f":
            colour = fill if fill is not None else stroke
        else:
            colour = stroke if stroke is not None else fill
        aci = rgb_to_aci(colour)

        closed_hint = bool(d.get("closePath", False)) or dtype in ("f", "fs")

        for points, closed in _items_to_contours(d.get("items", []), scale,
                                                 closed_hint):
            if len(points) >= 2:
                yield aci, {
                    "type": "polyline",
                    "points": points,
                    "closed": closed,
                }


# ---------------------------------------------------------------------------
# Document opening
# ---------------------------------------------------------------------------

def _open(filepath):
    """Open a PDF or AI file with PyMuPDF, raising PDFParseError on failure."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise PDFParseError(
            "PyMuPDF is not installed. Install it with:  pip install pymupdf"
        ) from e

    path = Path(filepath)
    if not path.exists():
        raise PDFParseError(f"File not found: {filepath}")

    ext = path.suffix.lower()
    try:
        if ext == ".ai":
            # Modern Illustrator files are PDF-compatible internally.
            doc = fitz.open(filepath, filetype="pdf")
        else:
            doc = fitz.open(filepath)
    except Exception as e:
        if ext == ".ai":
            raise PDFParseError(
                f"Could not read '{path.name}'. It appears to be an older "
                f"Illustrator file that is not PDF-compatible. Re-save it from "
                f"Illustrator with 'Create PDF Compatible File' enabled, or "
                f"export it as a PDF."
            ) from e
        raise PDFParseError(f"Could not read '{path.name}': {e}") from e

    if doc.page_count < 1:
        doc.close()
        raise PDFParseError(f"'{path.name}' contains no pages.")
    return doc


# ---------------------------------------------------------------------------
# Public API — mirrors dxf_parser
# ---------------------------------------------------------------------------

def _looks_pgf_only_ai(doc) -> bool:
    """True if this Illustrator file stores its artwork only in Adobe's
    proprietary private format, with no PDF-compatible page content.

    Such files were saved with 'Create PDF Compatible File' turned off and
    cannot be read by any tool other than Illustrator itself.
    """
    try:
        for x in range(1, doc.xref_length()):
            obj = doc.xref_object(x, compressed=True)
            if obj and ("AIPrivateData" in obj or "/Illustrator" in obj):
                return True
    except Exception:
        pass
    return False


def parse_pdf(filepath: str, color_speed_map: dict) -> dict:
    """Parse a PDF/AI file and return paths grouped by ACI colour.

    Mirrors ``dxf_parser.parse_dxf``: only colours present in
    ``color_speed_map`` are included in the result. Returns a dict of
    {aci_color: [list of path dicts]}.

    Raises PDFParseError for file-level problems or when the file contains no
    cuttable vector geometry at all.
    """
    doc = _open(filepath)
    color_paths: dict = {}
    total_found = 0
    pgf_only = False
    try:
        for page in doc:
            for aci, path in _extract_page_paths(page, PT_TO_MM):
                total_found += 1
                if aci in color_speed_map:
                    color_paths.setdefault(aci, []).append(path)
        if total_found == 0 and Path(filepath).suffix.lower() == ".ai":
            pgf_only = _looks_pgf_only_ai(doc)
    finally:
        doc.close()

    if total_found == 0:
        name = Path(filepath).name
        if pgf_only:
            raise PDFParseError(
                "'%s' was saved from Illustrator without PDF-compatible "
                "content, so its artwork cannot be read. In Illustrator, "
                "re-save it with 'Create PDF Compatible File' checked, or "
                "upload the design as a PDF or DXF instead." % name)
        raise PDFParseError(
            "No cuttable vector paths found in '%s'. The file may contain "
            "only a raster image (a photo or scan), or text that has not "
            "been converted to outlines." % name)
    return color_paths


def scan_pdf_colors(filepath: str) -> set:
    """Return the set of ACI colours present in a PDF/AI file."""
    doc = _open(filepath)
    colors = set()
    try:
        for page in doc:
            for aci, _path in _extract_page_paths(page, PT_TO_MM):
                if 1 <= aci <= 255:
                    colors.add(aci)
    finally:
        doc.close()
    return colors


def _bbox_mm(doc):
    """Bounding box (min_x, min_y, max_x, max_y) of all geometry, in mm."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for page in doc:
        for _aci, path in _extract_page_paths(page, PT_TO_MM):
            for x, y in path["points"]:
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y
    if min_x == float("inf"):
        return None
    return (min_x, min_y, max_x, max_y)


def pdf_dimensions(filepath: str):
    """Return (width_mm, height_mm) of the artwork bounding box."""
    doc = _open(filepath)
    try:
        box = _bbox_mm(doc)
    finally:
        doc.close()
    if box is None:
        return (0.0, 0.0)
    return (round(box[2] - box[0], 1), round(box[3] - box[1], 1))


def pdf_preview(filepath: str, output_path: str, dpi: int = 120):
    """Render a preview PNG of the first page and return (width_mm, height_mm).

    The render is cropped to the artwork bounding box (with a small margin) so
    the preview matches the DXF preview behaviour. Falls back to the full page
    if the bounding box cannot be determined.
    """
    import fitz
    doc = _open(filepath)
    try:
        page = doc[0]
        box = _bbox_mm(doc)
        if box is None and Path(filepath).suffix.lower() == ".ai" \
                and _looks_pgf_only_ai(doc):
            raise PDFParseError(
                "'%s' was saved from Illustrator without PDF-compatible "
                "content, so its artwork cannot be read. Re-save it with "
                "'Create PDF Compatible File' checked, or use a PDF or DXF."
                % Path(filepath).name)

        clip = None
        if box is not None:
            inv = 1.0 / PT_TO_MM  # mm -> points
            pad = 2.0 * inv       # 2 mm margin
            try:
                clip = fitz.Rect(
                    box[0] * inv - pad, box[1] * inv - pad,
                    box[2] * inv + pad, box[3] * inv + pad,
                ) & page.rect
                if clip.is_empty or clip.is_infinite:
                    clip = None
            except Exception:
                clip = None

        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                              alpha=False, clip=clip)
        pix.save(output_path)
    finally:
        doc.close()

    if box is None:
        return (0.0, 0.0)
    return (round(box[2] - box[0], 1), round(box[3] - box[1], 1))
