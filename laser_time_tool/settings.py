"""Parse job.txt files, map color names to ACI numbers, and motion constants."""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Motion / calibration constants — Ruida controller (from RDWorks 2026-02-25)
# ---------------------------------------------------------------------------

# Ruida idle speed (Cut params → Idle speed)
RAPID_SPEED_MM_S = 300.0

# Ruida Max Acc — optimized 2026-04-10 via grid search against 30 DXF files
# × 4 speeds (119 measurements). Lowered from 1600 to 1200 to improve
# 100 mm/s accuracy (88.9% → 90.7%) and overall (94.6% → 96.4%).
CUTTING_ACCEL_MM_S2 = 1200.0

# Ruida Idle Acc 2000 mm/s² × G0 Acc factor 120%
RAPID_ACCEL_MM_S2 = 2400.0

# Cornering tolerance — junction deviation model. Optimized 2026-04-10:
# raised from 0.005 to 0.006 for best overall accuracy (96.4%).
JUNCTION_DEVIATION_MM = 0.006

# Per-path fixed delay for laser on/off + controller command processing +
# settling time. Optimized 2026-04-10: reduced from 0.11 to 0.00. The old
# value over-estimated slow-speed cuts (20/16/12 mm/s) by 2.5–3.8%.
# Removing it improved slow speeds to 97.6–99.0% accuracy while the
# junction deviation and accel changes compensate at high speed.
PATH_OVERHEAD_S = 0.0

# Per-job fixed overhead: controller processing, file parsing, head
# positioning. Calibrated 2026-03-26 against 9 DXF files × 4 speeds
# (100/20/16/12 mm/s) vs Ruida — 0 under-estimates, mean err +2.2%.
JOB_OVERHEAD_S = 0.0

# ---------------------------------------------------------------------------
# Engraving (area-fill / raster scan) parameters — NEW in quote 1.5
# ---------------------------------------------------------------------------
# These are reasonable starting values for a Ruida-controlled machine. They
# are PENDING CALIBRATION against real shop test data and should be tuned
# once measured engraving times are available.

# Scan-line spacing (mm). Smaller = finer engraving and proportionally more time.
ENGRAVE_INTERVAL_MM = 0.1

# Head speed during an engraving sweep (mm/s).
ENGRAVE_SPEED_MM_S = 400.0

# Extra travel beyond each end of a scan-line so the head reaches full speed
# before the laser fires (one value, applied to each end).
ENGRAVE_OVERSCAN_MM = 0.0

# Effective head acceleration during engraving (mm/s^2). Calibrated 2026-05-21.
ENGRAVE_ACCEL_MM_S2 = 1200.0

# Fixed time per scan-line and per job (s). Calibrated 2026-05-21.
ENGRAVE_LINE_OVERHEAD_S = 0.0
ENGRAVE_JOB_OVERHEAD_S = 22.0

# AutoCAD Color Index (ACI) standard color names
COLOR_NAME_TO_ACI = {
    "red": 1,
    "yellow": 2,
    "green": 3,
    "cyan": 4,
    "blue": 5,
    "magenta": 6,
    "white": 7,
    "black": 7,
}

ACI_TO_NAME = {
    1: "Red",
    2: "Yellow",
    3: "Green",
    4: "Cyan",
    5: "Blue",
    6: "Magenta",
    7: "White",
}


def parse_job_txt(job_path: Path) -> tuple[str, dict[int, float]]:
    """
    Parse a job.txt file and return material name + color-speed mapping.

    job.txt format:
        material: 3mm Clear Acrylic
        red 10
        green 100
        blue 300

    Returns:
        (material_name, {aci_number: speed_mm_s})
    """
    material = ""
    color_speed_map = {}

    text = Path(job_path).read_text(encoding="utf-8")

    for line in text.splitlines():
        line = line.strip()

        # Skip blank lines and comments
        if not line or line.startswith("#"):
            continue

        # Check for material line
        if line.lower().startswith("material"):
            # Strip "material" prefix and separator
            value = re.sub(r"^material\s*[:=]?\s*", "", line, flags=re.IGNORECASE)
            material = value.strip()
            continue

        # Parse color-speed pairs
        # Supports: "red 10", "red=10", "red: 10", "red\t10"
        parts = re.split(r"[\s=:]+", line, maxsplit=1)
        if len(parts) != 2:
            continue

        color_key = parts[0].strip().lower()
        speed_str = parts[1].strip()

        try:
            speed = float(speed_str)
        except ValueError:
            continue

        # Accept named colors ("red") or numeric ACI codes ("250")
        if color_key in COLOR_NAME_TO_ACI:
            aci = COLOR_NAME_TO_ACI[color_key]
        elif color_key.isdigit() and 1 <= int(color_key) <= 255:
            aci = int(color_key)
        else:
            continue

        color_speed_map[aci] = speed

    return material, color_speed_map


def resolve_entity_color(entity, doc) -> int:
    """
    Resolve the ACI color of a DXF entity.

    Priority:
    1. Direct ACI color (1-255)
    2. BYLAYER (256) -> use layer color
    3. BYBLOCK (0) -> return 0 (caller should warn)
    """
    color = entity.dxf.get("color", 256)  # Default is BYLAYER

    if 1 <= color <= 255:
        return color

    if color == 256:  # BYLAYER
        layer_name = entity.dxf.get("layer", "0")
        try:
            layer = doc.layers.get(layer_name)
            layer_color = layer.color
            if 1 <= layer_color <= 255:
                return layer_color
        except Exception:
            pass
        return 7  # Default to white if layer lookup fails

    if color == 0:  # BYBLOCK
        return 0

    return 7  # Fallback
