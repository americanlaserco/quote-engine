"""Render DXF files to PNG images for Groq vision analysis.

Uses ezdxf + matplotlib to produce clean, high-contrast renderings
of DXF geometry that vision models can analyze for spatial features.
"""

import math
import os
from pathlib import Path

import ezdxf
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection


def render_dxf_to_png(
    dxf_path: str,
    output_path: str = None,
    dpi: int = 150,
    figsize: tuple = (10, 10),
    color_by_layer: bool = True,
) -> str:
    """
    Render a DXF file to a PNG image.

    Args:
        dxf_path: Path to the DXF file
        output_path: Where to save the PNG. If None, saves next to DXF as .png
        dpi: Resolution (150 is good for vision models — clear but not huge)
        figsize: Figure size in inches
        color_by_layer: If True, color entities by their DXF color

    Returns:
        Path to the saved PNG file
    """
    if output_path is None:
        output_path = str(Path(dxf_path).with_suffix(".png"))

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor("white")
    ax.set_aspect("equal")

    # ACI color mapping for rendering
    aci_colors = {
        0: "#000000", 1: "#FF0000", 2: "#FFFF00", 3: "#00FF00",
        4: "#00FFFF", 5: "#0000FF", 6: "#FF00FF", 7: "#000000",
    }

    lines_by_color = {}

    for entity in msp:
        # Resolve color
        color_idx = entity.dxf.color if hasattr(entity.dxf, "color") else 7
        if color_idx == 256:  # BYLAYER
            layer = doc.layers.get(entity.dxf.layer) if hasattr(entity.dxf, "layer") else None
            color_idx = layer.color if layer else 7
        color = aci_colors.get(color_idx, "#333333")

        if color not in lines_by_color:
            lines_by_color[color] = []

        etype = entity.dxftype()

        if etype == "LINE":
            sx, sy = entity.dxf.start.x, entity.dxf.start.y
            ex, ey = entity.dxf.end.x, entity.dxf.end.y
            lines_by_color[color].append([(sx, sy), (ex, ey)])

        elif etype == "LWPOLYLINE":
            pts = list(entity.get_points(format="xyb"))
            for i in range(len(pts) - 1):
                x1, y1, b1 = pts[i]
                x2, y2, _ = pts[i + 1]
                if abs(b1) > 1e-6:
                    arc_pts = _bulge_to_points(x1, y1, x2, y2, b1)
                    for j in range(len(arc_pts) - 1):
                        lines_by_color[color].append([arc_pts[j], arc_pts[j + 1]])
                else:
                    lines_by_color[color].append([(x1, y1), (x2, y2)])
            if entity.closed and len(pts) > 1:
                x1, y1, b1 = pts[-1]
                x2, y2, _ = pts[0]
                if abs(b1) > 1e-6:
                    arc_pts = _bulge_to_points(x1, y1, x2, y2, b1)
                    for j in range(len(arc_pts) - 1):
                        lines_by_color[color].append([arc_pts[j], arc_pts[j + 1]])
                else:
                    lines_by_color[color].append([(x1, y1), (x2, y2)])

        elif etype == "CIRCLE":
            cx, cy = entity.dxf.center.x, entity.dxf.center.y
            r = entity.dxf.radius
            n = max(36, int(2 * math.pi * r / 0.5))
            for i in range(n):
                a1 = 2 * math.pi * i / n
                a2 = 2 * math.pi * (i + 1) / n
                lines_by_color[color].append([
                    (cx + r * math.cos(a1), cy + r * math.sin(a1)),
                    (cx + r * math.cos(a2), cy + r * math.sin(a2)),
                ])

        elif etype == "ARC":
            cx, cy = entity.dxf.center.x, entity.dxf.center.y
            r = entity.dxf.radius
            start_deg = entity.dxf.start_angle
            end_deg = entity.dxf.end_angle
            sweep = end_deg - start_deg
            if sweep < 0:
                sweep += 360
            n = max(8, int(sweep / 5))
            for i in range(n):
                a1 = math.radians(start_deg + sweep * i / n)
                a2 = math.radians(start_deg + sweep * (i + 1) / n)
                lines_by_color[color].append([
                    (cx + r * math.cos(a1), cy + r * math.sin(a1)),
                    (cx + r * math.cos(a2), cy + r * math.sin(a2)),
                ])

        elif etype == "SPLINE":
            try:
                pts = list(entity.flattening(0.1))
                for i in range(len(pts) - 1):
                    lines_by_color[color].append([
                        (pts[i].x, pts[i].y),
                        (pts[i + 1].x, pts[i + 1].y),
                    ])
            except Exception:
                pass

        elif etype == "ELLIPSE":
            try:
                pts = list(entity.flattening(0.1))
                for i in range(len(pts) - 1):
                    lines_by_color[color].append([
                        (pts[i].x, pts[i].y),
                        (pts[i + 1].x, pts[i + 1].y),
                    ])
            except Exception:
                pass

    # Draw all lines grouped by color
    for color, segs in lines_by_color.items():
        if segs:
            lc = LineCollection(segs, colors=color, linewidths=0.8)
            ax.add_collection(lc)

    ax.autoscale()
    ax.margins(0.05)
    ax.set_axis_off()

    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)

    return output_path


def _bulge_to_points(x1, y1, x2, y2, bulge, n_segments=16):
    """Convert a polyline bulge segment to a list of (x, y) points."""
    dx = x2 - x1
    dy = y2 - y1
    chord = math.sqrt(dx * dx + dy * dy)
    if chord < 1e-9:
        return [(x1, y1), (x2, y2)]

    sagitta = abs(bulge) * chord / 2
    radius = ((chord / 2) ** 2 + sagitta ** 2) / (2 * sagitta)

    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    d = radius - sagitta
    nx = -dy / chord
    ny = dx / chord

    if bulge > 0:
        cx = mx + d * nx
        cy = my + d * ny
    else:
        cx = mx - d * nx
        cy = my - d * ny

    a1 = math.atan2(y1 - cy, x1 - cx)
    a2 = math.atan2(y2 - cy, x2 - cx)

    if bulge > 0:  # CCW
        if a2 < a1:
            a2 += 2 * math.pi
    else:  # CW
        if a1 < a2:
            a1 += 2 * math.pi

    points = []
    for i in range(n_segments + 1):
        t = i / n_segments
        a = a1 + (a2 - a1) * t
        points.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return points


def render_all_dxfs_in_folder(folder: str, output_folder: str = None) -> list[str]:
    """Render all DXF files in a folder to PNGs. Returns list of PNG paths."""
    folder = Path(folder)
    if output_folder is None:
        output_folder = folder / "_renders"
    else:
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    png_paths = []
    for dxf_path in sorted(folder.glob("*.dxf")):
        out = output_folder / f"{dxf_path.stem}.png"
        try:
            render_dxf_to_png(str(dxf_path), str(out))
            png_paths.append(str(out))
            print(f"  Rendered: {dxf_path.name} -> {out.name}")
        except Exception as e:
            print(f"  ERROR rendering {dxf_path.name}: {e}")
    return png_paths


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "test files"
    render_all_dxfs_in_folder(folder)
