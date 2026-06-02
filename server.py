"""
Laser Cut Quoting Tool - Web Interface
"""
import base64, io, json, math, os, sys, tempfile, time, traceback
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from laser_time_tool.dxf_parser import parse_dxf, path_length
from laser_time_tool.motion_planner import estimate_time_offline
from laser_time_tool.calibration import apply_calibration
from laser_time_tool.engrave_planner import estimate_engrave_time
from laser_time_tool import engrave_calibration
from laser_time_tool.settings import (
    parse_job_txt, COLOR_NAME_TO_ACI, ACI_TO_NAME, resolve_entity_color,
    RAPID_SPEED_MM_S, CUTTING_ACCEL_MM_S2, RAPID_ACCEL_MM_S2,
    JUNCTION_DEVIATION_MM, PATH_OVERHEAD_S, JOB_OVERHEAD_S,
    ENGRAVE_INTERVAL_MM, ENGRAVE_SPEED_MM_S, ENGRAVE_OVERSCAN_MM,
    ENGRAVE_ACCEL_MM_S2, ENGRAVE_JOB_OVERHEAD_S,
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
debug_logs = []


def _scan_dxf_colors(dxf_paths: list) -> set:
    """Scan DXF files and return all unique ACI colors found."""
    import ezdxf
    from laser_time_tool.dxf_parser import _iter_entities
    colors = set()
    for path in dxf_paths:
        try:
            doc = ezdxf.readfile(path)
            msp = doc.modelspace()
            for entity in _iter_entities(msp, doc):
                try:
                    aci = resolve_entity_color(entity, doc)
                    if 1 <= aci <= 255:
                        colors.add(aci)
                except Exception:
                    pass
        except Exception:
            pass
    return colors


# Vector input formats the engine accepts.
VECTOR_EXTS = (".dxf", ".pdf", ".ai")


def _is_pdf_like(path) -> bool:
    """True for PDF / Adobe Illustrator files (read natively, not via ezdxf)."""
    return str(path).lower().endswith((".pdf", ".ai"))


def parse_vector(path, color_speed_map):
    """Parse a DXF, PDF, or AI file into {aci: [paths]} via the right reader.

    DXF files use the ezdxf-based parser. PDF and AI files use the native
    PyMuPDF-based reader (added in quote 1.4). Both return the same path-dict
    format, so the rest of the pipeline is identical.
    """
    if _is_pdf_like(path):
        from laser_time_tool.pdf_parser import parse_pdf
        return parse_pdf(path, color_speed_map)
    return parse_dxf(path, color_speed_map)


def _scan_vector_colors(paths: list) -> set:
    """Scan DXF/PDF/AI files and return all unique ACI colors found."""
    colors = set()
    dxfs = [p for p in paths if not _is_pdf_like(p)]
    pdfs = [p for p in paths if _is_pdf_like(p)]
    if dxfs:
        colors |= _scan_dxf_colors(dxfs)
    if pdfs:
        from laser_time_tool.pdf_parser import scan_pdf_colors
        for p in pdfs:
            try:
                colors |= scan_pdf_colors(p)
            except Exception as e:
                log("Color scan failed for %s: %s" % (os.path.basename(p), e), "WARN")
    return colors


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    entry = "[%s] [%s] %s" % (ts, level, msg)
    debug_logs.append(entry)
    print(entry)

def fmt(seconds):
    t = int(round(seconds))
    if t < 60:
        return "%ds" % t
    m, s = divmod(t, 60)
    if m < 60:
        return "%dm %02ds" % (m, s)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)

MATERIAL_PRESETS = {
    "paper_cardstock": {"name": "Paper / Cardstock", "cut_speed": 100, "description": "Standard paper, cardstock, chipboard"},
    "3mm_acrylic": {"name": "3mm Acrylic", "cut_speed": 16, "description": "3mm cast acrylic"},
    "6mm_acrylic": {"name": "6mm Acrylic", "cut_speed": 11, "description": "6mm cast acrylic"},
    "3mm_birch_ply": {"name": "3mm Birch Plywood", "cut_speed": 20, "description": "3mm Baltic birch"},
    "6mm_birch_ply": {"name": "6mm Birch Plywood", "cut_speed": 16, "description": "6mm Baltic birch"},
    "felt_fabric": {"name": "Felt / Fabric", "cut_speed": 100, "description": "Felt, cotton, polyester"},
    "oil_board": {"name": "Oil Board", "cut_speed": 100, "description": "Oil board paper 24x36"},
    "custom": {"name": "Custom", "cut_speed": 100, "description": "Set your own speeds"},
}

@app.route("/")
def index():
    return send_from_directory(str(project_root), "frontend.html")

@app.route("/api/defaults", methods=["GET"])
def get_defaults():
    return jsonify({"rapid_speed": RAPID_SPEED_MM_S, "cutting_accel": CUTTING_ACCEL_MM_S2,
        "rapid_accel": RAPID_ACCEL_MM_S2, "junction_deviation": JUNCTION_DEVIATION_MM,
        "path_overhead": PATH_OVERHEAD_S, "job_overhead": JOB_OVERHEAD_S, "default_rate": 2.10, "default_minimum": 65.0,
        "colors": dict(COLOR_NAME_TO_ACI), "aci_names": {str(k): v for k, v in ACI_TO_NAME.items()}})

@app.route("/api/materials", methods=["GET"])
def get_materials():
    return jsonify(MATERIAL_PRESETS)

@app.route("/api/preview", methods=["POST"])
def preview_dxf():
    f = request.files.get("dxf_file")
    if not f or not f.filename:
        return jsonify({"error": "No file"}), 400
    tmpdir = tempfile.mkdtemp(prefix="laserpreview_")
    try:
        src_path = os.path.join(tmpdir, f.filename)
        f.save(src_path)
        png_path = os.path.join(tmpdir, "preview.png")
        ext = os.path.splitext(f.filename)[1].lower()

        if ext in (".pdf", ".ai"):
            # PDF / Illustrator - read natively with PyMuPDF.
            from laser_time_tool.pdf_parser import pdf_preview
            width_mm, height_mm = pdf_preview(src_path, png_path, dpi=120)
            native_unit = "mm"  # PDF/AI carry no native design unit
            log("Preview %s: %s, bbox=%.1f x %.1f mm" % (
                f.filename, ext.lstrip(".").upper(), width_mm, height_mm))
        else:
            # DXF - render with ezdxf + matplotlib.
            from laser_time_tool.dxf_renderer import render_dxf_to_png
            render_dxf_to_png(src_path, png_path, dpi=120, figsize=(10, 10))
            import ezdxf
            doc = ezdxf.readfile(src_path)
            msp = doc.modelspace()
            insunits = doc.header.get("$INSUNITS", 0) if hasattr(doc, "header") else 0
            unit_scale = {0: 1.0, 1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0}.get(insunits, 1.0)
            unit_name = {0: "unspecified", 1: "inches", 2: "feet", 4: "mm", 5: "cm", 6: "m"}.get(insunits, "unknown")
            native_unit = "in" if insunits in (1, 2) else "mm"
            from ezdxf import bbox as ezdxf_bbox
            cache = ezdxf_bbox.Cache()
            box = ezdxf_bbox.extents(msp, cache=cache)
            if box.has_data:
                width_mm = round(box.size.x * unit_scale, 1)
                height_mm = round(box.size.y * unit_scale, 1)
            else:
                width_mm = 0
                height_mm = 0
            log("Preview %s: units=%s (scale=%.1f), bbox=%.1f x %.1f mm" % (
                f.filename, unit_name, unit_scale, width_mm, height_mm))

        with open(png_path, "rb") as pf:
            b64 = base64.b64encode(pf.read()).decode("utf-8")
        return jsonify({"image": "data:image/png;base64," + b64, "filename": f.filename,
            "width_mm": width_mm, "height_mm": height_mm, "unit": native_unit})
    except Exception as e:
        return jsonify({"error": str(e), "filename": f.filename}), 500
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

def _quote_one_file(dxf_path, color_speed_map, rapid_speed, cutting_accel,
                    rapid_accel, junction_dev, path_overhead, job_overhead, rate,
                    engrave_acis=None, engrave_params=None):
    """Quote a single DXF/PDF/AI file. Returns a result dict.

    Cut and score colours go through the motion planner. Colours listed in
    engrave_acis are area-filled by the engraving planner instead.
    """
    engrave_acis = engrave_acis or set()
    engrave_params = engrave_params or {}
    fname = os.path.basename(dxf_path)
    stem = Path(dxf_path).stem
    log("Processing: %s" % fname)
    color_paths = parse_vector(dxf_path, color_speed_map)

    # Split colours into cut/score paths and engrave (area-fill) paths.
    cut_paths = {}
    engrave_paths = []
    for aci, paths in color_paths.items():
        if aci in engrave_acis:
            engrave_paths.extend(paths)
        else:
            cut_paths[aci] = paths

    entities = sum(len(p) for p in color_paths.values())
    total_cut_mm = sum(path_length(p) for paths in cut_paths.values() for p in paths)
    log("  Entities: %d (cut/score: %d, engrave: %d), cut length: %.1f mm" % (
        entities, sum(len(p) for p in cut_paths.values()),
        len(engrave_paths), total_cut_mm))
    for aci, paths in cut_paths.items():
        cn = ACI_TO_NAME.get(aci, "ACI%d" % aci)
        sp = color_speed_map.get(aci, 0)
        pl = sum(path_length(p) for p in paths)
        log("  %s: %d paths, %.1f mm @ %s mm/s" % (cn, len(paths), pl, sp))

    # --- cut / score time: motion planner + per-file calibration ---
    if cut_paths:
        raw_cut = estimate_time_offline(cut_paths, color_speed_map,
            rapid_speed=rapid_speed, acceleration=cutting_accel,
            rapid_acceleration=rapid_accel, junction_deviation=junction_dev,
            path_overhead=path_overhead, job_overhead=job_overhead)
        speed_for_cal = next((color_speed_map[a] for a in cut_paths), 100.0)
        cut_entities = sum(len(p) for p in cut_paths.values())
        # Only pass source_path for DXF — the danger detector inspects DXF
        # entities and can't read PDF/AI files. PDF/AI fall back to the
        # geometry-only model (still safe, just no risk surcharge).
        src_for_detector = dxf_path if str(dxf_path).lower().endswith('.dxf') else None
        calibrated_cut = apply_calibration(raw_cut, stem, speed_for_cal,
            num_paths=cut_entities, total_cut_mm=total_cut_mm,
            source_path=src_for_detector)
    else:
        raw_cut = 0.0
        calibrated_cut = 0.0
    cal_offset = calibrated_cut - raw_cut
    log("  Cut/score: raw %s -> calibrated %s (offset %+.1fs)" % (
        fmt(raw_cut), fmt(calibrated_cut), cal_offset))

    # --- engrave time: area-fill scan planner + feedback calibration ---
    engrave_seconds = 0.0
    engrave_calibrated = False
    eng_stats = {"scanlines": 0, "engraved_area_mm2": 0.0,
                 "swept_length_mm": 0.0, "fill_height_mm": 0.0}
    if engrave_paths:
        eng_speed = engrave_params.get("speed", ENGRAVE_SPEED_MM_S)
        raw_engrave, eng_stats = estimate_engrave_time(
            engrave_paths,
            interval=engrave_params.get("interval", ENGRAVE_INTERVAL_MM),
            speed=eng_speed,
            overscan=engrave_params.get("overscan", ENGRAVE_OVERSCAN_MM),
            accel=engrave_params.get("accel", ENGRAVE_ACCEL_MM_S2),
            job_overhead=engrave_params.get("job_overhead", ENGRAVE_JOB_OVERHEAD_S))
        engrave_seconds = engrave_calibration.apply_calibration(
            raw_engrave, stem, eng_speed)
        engrave_calibrated = engrave_calibration.is_measured_design(stem)
        log("  Engrave: %s%s, %d scan lines, %.0f mm2 @ %s mm/s, %s mm interval" % (
            fmt(engrave_seconds),
            " (design-calibrated)" if engrave_calibrated else "",
            eng_stats["scanlines"], eng_stats["engraved_area_mm2"],
            eng_speed, engrave_params.get("interval", ENGRAVE_INTERVAL_MM)))

    total_seconds = calibrated_cut + engrave_seconds
    raw_total = raw_cut + engrave_seconds
    cost = total_seconds / 60.0 * rate
    log("  Total: %s  Cost: $%.2f" % (fmt(total_seconds), cost))

    color_bd = []
    for aci, paths in color_paths.items():
        name = ACI_TO_NAME.get(aci, "ACI%d" % aci)
        if aci in engrave_acis:
            color_bd.append({"color": name, "aci": aci, "mode": "engrave",
                "paths": len(paths), "length_mm": 0.0,
                "speed": engrave_params.get("speed", ENGRAVE_SPEED_MM_S),
                "engrave_area_mm2": eng_stats["engraved_area_mm2"]})
        else:
            color_bd.append({"color": name, "aci": aci, "mode": "cut",
                "paths": len(paths),
                "length_mm": round(sum(path_length(p) for p in paths), 1),
                "speed": color_speed_map.get(aci, 0)})

    return {"filename": fname, "stem": stem, "entities": entities,
        "total_cut_mm": round(total_cut_mm, 1), "raw_seconds": round(raw_total, 2),
        "raw_time": fmt(raw_total), "calibration_offset": round(cal_offset, 1),
        "cut_seconds": round(calibrated_cut, 2),
        "engrave_seconds": round(engrave_seconds, 2),
        "engrave_area_mm2": eng_stats["engraved_area_mm2"],
        "engrave_scanlines": eng_stats["scanlines"],
        "engrave_calibrated": engrave_calibrated,
        "seconds": round(total_seconds, 2), "time": fmt(total_seconds),
        "cost": round(cost, 2), "color_breakdown": color_bd, "status": "ok"}

@app.route("/api/quote", methods=["POST"])
def run_quote():
    global debug_logs
    debug_logs = []
    start = time.time()
    log("=== LASER CUT QUOTE - Starting ===")
    try:
        settings = json.loads(request.form.get("settings", "{}"))
        log("Settings: %s" % json.dumps(settings))
        # SECURITY: pricing is server-side only. Client-supplied "rate" and
        # "minimum" are honored ONLY if the request carries the admin token
        # that matches ADMIN_PIN (an env var, never sent to the client).
        # Anonymous public requests always use the server-configured values
        # so a hacked browser can't quote at $0.01/min.
        _server_rate = float(os.environ.get("SHOP_RATE", "2.10"))
        _server_min  = float(os.environ.get("SHOP_MINIMUM", "65.00"))
        _admin_pin   = os.environ.get("ADMIN_PIN", "")
        _client_pin  = request.headers.get("X-Admin-Pin", "")
        _is_admin    = bool(_admin_pin and _client_pin and _admin_pin == _client_pin)
        if _is_admin:
            rate = float(settings.get("rate", _server_rate))
            minimum = float(settings.get("minimum", _server_min))
        else:
            rate = _server_rate
            minimum = _server_min
        rapid_speed = float(settings.get("rapid_speed", RAPID_SPEED_MM_S))
        cutting_accel = float(settings.get("cutting_accel", CUTTING_ACCEL_MM_S2))
        rapid_accel = float(settings.get("rapid_accel", RAPID_ACCEL_MM_S2))
        junction_dev = float(settings.get("junction_deviation", JUNCTION_DEVIATION_MM))
        path_overhead = float(settings.get("path_overhead", PATH_OVERHEAD_S))
        job_overhead = float(settings.get("job_overhead", JOB_OVERHEAD_S))
        tmpdir = tempfile.mkdtemp(prefix="laserquote_")
        dxf_files = request.files.getlist("dxf_files[]")
        if not dxf_files or all(f.filename == "" for f in dxf_files):
            return jsonify({"error": "No files uploaded", "debug_log": "\n".join(debug_logs)}), 400
        saved = []
        for f in dxf_files:
            if f.filename and f.filename.lower().endswith(VECTOR_EXTS):
                fp = os.path.join(tmpdir, f.filename)
                f.save(fp)
                saved.append(fp)
                log("Saved: %s" % f.filename)
        if not saved:
            return jsonify({"error": "No DXF, AI, or PDF files in upload", "debug_log": "\n".join(debug_logs)}), 400
        color_speed_map = {}
        material = settings.get("material", "")
        job_file = request.files.get("job_txt")
        if job_file and job_file.filename:
            jp = os.path.join(tmpdir, "job.txt")
            job_file.save(jp)
            material, color_speed_map = parse_job_txt(Path(jp))
            log("job.txt -> material=%s, csm=%s" % (material, color_speed_map))
        cs = settings.get("color_speeds", {})
        for cn, sp in cs.items():
            try:
                sp = float(sp)
                if sp <= 0: continue
                if cn.lower() in COLOR_NAME_TO_ACI: aci = COLOR_NAME_TO_ACI[cn.lower()]
                elif cn.isdigit(): aci = int(cn)
                else: continue
                color_speed_map[aci] = sp
            except (ValueError, TypeError): pass
        if not color_speed_map:
            ds = float(settings.get("default_speed", 100))
            found_colors = _scan_vector_colors(saved)
            if found_colors:
                color_speed_map = {c: ds for c in found_colors}
            else:
                color_speed_map = {i: ds for i in range(1, 256)}
            log("Default speed %s mm/s for %d colors: %s" % (ds, len(color_speed_map), sorted(color_speed_map.keys())))
        # --- engraving configuration (quote 1.5) ---
        engrave_acis = set()
        engrave_color = str(settings.get("engrave_color", "")).strip().lower()
        if engrave_color and engrave_color not in ("none", "off"):
            if engrave_color in COLOR_NAME_TO_ACI:
                engrave_acis.add(COLOR_NAME_TO_ACI[engrave_color])
            elif engrave_color.isdigit():
                engrave_acis.add(int(engrave_color))
        eng_g_accel, eng_g_joh = engrave_calibration.get_global_params()
        engrave_params = {
            "interval": float(settings.get("engrave_interval", ENGRAVE_INTERVAL_MM)),
            "speed": float(settings.get("engrave_speed", ENGRAVE_SPEED_MM_S)),
            "overscan": float(settings.get("engrave_overscan", ENGRAVE_OVERSCAN_MM)),
            "accel": eng_g_accel,
            "job_overhead": eng_g_joh,
        }
        # Ensure engrave colours survive parsing even with no cut speed set.
        for aci in engrave_acis:
            color_speed_map.setdefault(aci, engrave_params["speed"])
        if engrave_acis:
            log("Engraving ON: colors=%s, interval=%s mm, speed=%s mm/s, overscan=%s mm" % (
                sorted(engrave_acis), engrave_params["interval"],
                engrave_params["speed"], engrave_params["overscan"]))
        results = []
        total_seconds = 0.0
        for dxf_path in sorted(saved):
            try:
                r = _quote_one_file(dxf_path, color_speed_map, rapid_speed, cutting_accel, rapid_accel, junction_dev, path_overhead, job_overhead, rate, engrave_acis, engrave_params)
                total_seconds += r["seconds"]
                results.append(r)
            except Exception as e:
                log("ERROR on %s: %s" % (os.path.basename(dxf_path), e), "ERROR")
                log(traceback.format_exc(), "ERROR")
                results.append({"filename": os.path.basename(dxf_path), "stem": Path(dxf_path).stem, "status": "error", "error": str(e), "seconds": 0, "time": "ERROR", "cost": 0, "entities": 0})
        total_engrave_seconds = sum(r.get("engrave_seconds", 0) for r in results)
        total_cut_seconds = sum(r.get("cut_seconds", 0) for r in results)
        total_cost = total_seconds / 60.0 * rate
        elapsed = time.time() - start
        below_minimum = bool(total_cost < minimum)
        final_cost = max(total_cost, minimum)
        log("=== COMPLETE: %d files, %s, $%.2f%s in %.2fs ===" % (
            len(results), fmt(total_seconds), final_cost,
            " (minimum)" if below_minimum else "", elapsed))
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        return jsonify({"results": results,
            "summary": {"total_files": len(results), "total_seconds": round(total_seconds, 2),
                "total_time": fmt(total_seconds), "unit_cost": round(total_cost, 2),
                "total_cut_seconds": round(total_cut_seconds, 2),
                "total_engrave_seconds": round(total_engrave_seconds, 2),
                "total_cost": round(final_cost, 2), "below_minimum": below_minimum,
                "minimum": minimum,
                "rate": rate, "material": material, "processing_time": round(elapsed, 2)},
            "settings_used": {"rapid_speed": rapid_speed, "cutting_accel": cutting_accel,
                "rapid_accel": rapid_accel, "junction_deviation": junction_dev,
                "path_overhead": path_overhead, "job_overhead": job_overhead, "rate": rate,
                "color_speed_map": {str(k): v for k, v in color_speed_map.items()}},
            "debug_log": "\n".join(debug_logs)})
    except Exception as e:
        log("FATAL: %s" % e, "ERROR")
        log(traceback.format_exc(), "ERROR")
        return jsonify({"error": str(e), "debug_log": "\n".join(debug_logs)}), 500

@app.route("/api/log-engrave-actual", methods=["POST"])
def log_engrave_actual():
    """Record a real measured engrave time and re-fit the calibration.

    The shop uploads the design, the engrave colour, the speed used and the
    actual machine time. The model is re-fitted from every logged job, so the
    estimate gets more accurate the more jobs are recorded.
    """
    try:
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"error": "No file uploaded"}), 400
        try:
            actual = float(request.form.get("actual_seconds", 0))
            speed = float(request.form.get("speed", 0))
        except (TypeError, ValueError):
            return jsonify({"error": "Actual time and speed must be numbers"}), 400
        try:
            interval = float(request.form.get("interval") or ENGRAVE_INTERVAL_MM)
        except (TypeError, ValueError):
            interval = ENGRAVE_INTERVAL_MM
        engrave_color = str(request.form.get("engrave_color", "")).strip().lower()
        if actual <= 0 or speed <= 0:
            return jsonify({"error": "Enter a positive actual time and engrave speed"}), 400
        aci = COLOR_NAME_TO_ACI.get(engrave_color)
        if aci is None:
            return jsonify({"error": "Choose the engrave colour used for this job"}), 400
        tmpdir = tempfile.mkdtemp(prefix="englog_")
        try:
            fp_in = os.path.join(tmpdir, f.filename)
            f.save(fp_in)
            cp = parse_vector(fp_in, {aci: 100.0})
            paths = cp.get(aci, [])
            if not paths:
                return jsonify({"error": "No %s engrave geometry found in that file" % engrave_color}), 400
            hist, nlines = engrave_calibration.compute_extents(paths, interval)
            stats = engrave_calibration.record_measurement(
                Path(f.filename).stem, speed, interval, actual, hist, nlines)
            log("Engrave actual logged: %s @ %s mm/s = %.0fs (now %s measurements, model err %s%%)" % (
                f.filename, speed, actual, stats.get("n_measurements"),
                stats.get("model_err_pct")))
            return jsonify({"success": True, "stats": stats})
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        log("Engrave log error: %s" % e, "ERROR")
        return jsonify({"error": str(e)}), 500


@app.route("/api/laser-last-job-time", methods=["GET"])
def laser_last_job_time():
    """Read the last completed job's duration from a connected Ruida controller.

    Only works on a machine on the same network or USB as the laser. The Ruida
    protocol is reverse-engineered \u2014 check the value against the controller
    panel the first time.
    """
    try:
        from laser_time_tool.ruida_connection import (
            RuidaConnection, RuidaSerialConnection, discover_ruida,
            find_ruida_serial, MEM_PREV_WORK_TIME)
        ip = (request.args.get("ip", "") or "").strip()
        tried = []
        for addr in ([ip] if ip else discover_ruida()):
            tried.append("network " + addr)
            try:
                with RuidaConnection(addr, timeout=2.0) as conn:
                    t = conn.get_prev_work_time()
                    if t:
                        return jsonify({"success": True, "seconds": float(t),
                                        "source": "Ethernet " + addr})
            except Exception:
                pass
        port = find_ruida_serial()
        if port:
            tried.append("USB " + port)
            try:
                with RuidaSerialConnection(port=port) as conn:
                    t = conn.query_value(MEM_PREV_WORK_TIME)
                    if t:
                        return jsonify({"success": True, "seconds": float(t),
                                        "source": "USB " + port})
            except Exception:
                pass
        return jsonify({"success": False,
                        "error": "No Ruida controller reachable (%s). This must "
                                 "run on a PC connected to the laser." %
                                 (", ".join(tried) if tried else "none found")})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/email-quote", methods=["POST"])
def email_quote():
    """Send quote summary to client and shop owner."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    try:
        data = request.get_json()
        client_email = data.get("client_email", "").strip()
        if not client_email or "@" not in client_email:
            return jsonify({"success": False, "error": "Invalid email address"}), 400

        quote = data.get("quote_data", {})
        results = quote.get("results", [])
        summary = quote.get("summary", {})
        qty = data.get("quantity", 1)
        cutting_total = data.get("cutting_total", 0)
        material_cost = data.get("material_cost", 0)
        grand_total = data.get("grand_total", 0)
        material_name = data.get("material_name", "")
        own_material = data.get("own_material", False)

        lines = []
        lines.append("AMERICAN LASER CO. - QUOTE ESTIMATE")
        lines.append("=" * 45)
        lines.append("")
        lines.append("** This is a beta estimate. Final pricing")
        lines.append("   requires review and may differ. **")
        lines.append("")
        if material_name:
            lines.append("Material: %s%s" % (material_name, " (customer provided)" if own_material else ""))
        lines.append("Quantity: %d" % qty)
        lines.append("")
        lines.append("%-22s %8s %8s" % ("File", "Entities", "Cost"))
        lines.append("-" * 40)
        for r in results:
            if r.get("status") == "error":
                lines.append("%-22s %8s %8s" % (r.get("stem", r.get("filename", "?")), "ERR", "--"))
            else:
                lines.append("%-22s %8d %8s" % (r.get("stem", ""), r.get("entities", 0), "$%.2f" % r.get("cost", 0)))
        lines.append("-" * 40)
        lines.append("%-22s %8s %8s" % ("Cutting total", "", "$%.2f" % cutting_total))
        if material_cost > 0:
            lines.append("%-22s %8s %8s" % ("Material", "", "$%.2f" % material_cost))
        lines.append("")
        lines.append("%-22s %8s %8s" % ("ESTIMATED TOTAL", "", "$%.2f" % grand_total))
        lines.append("")
        lines.append("---")
        lines.append("This estimate was generated by the American Laser Co.")
        lines.append("digital quoting tool (beta). For questions or to")
        lines.append("confirm your order, reply to this email or call us.")
        lines.append("")
        lines.append("American Laser Co.")
        lines.append("americanlaserco@gmail.com")

        body_text = "\n".join(lines)

        file_rows = ""
        for r in results:
            if r.get("status") == "error":
                file_rows += "<tr><td>%s</td><td style='text-align:right'>ERR</td><td style='text-align:right'>--</td></tr>" % r.get("stem", "?")
            else:
                file_rows += "<tr><td>%s</td><td style='text-align:right'>%d</td><td style='text-align:right'>$%.2f</td></tr>" % (r.get("stem", ""), r.get("entities", 0), r.get("cost", 0))

        mat_row = ""
        if material_cost > 0:
            mat_row = "<tr><td><strong>Material</strong></td><td></td><td style='text-align:right'><strong>$%.2f</strong></td></tr>" % material_cost

        body_html = """<div style="font-family:Arial,sans-serif;max-width:500px;margin:0 auto;color:#333">
<h2 style="color:#ff4d2d;margin-bottom:4px">American Laser Co.</h2>
<p style="color:#888;font-size:13px;margin-top:0">Quote Estimate</p>
<div style="background:#fff3cd;border:1px solid #ffc107;border-radius:6px;padding:10px 14px;margin:12px 0;font-size:13px;color:#856404">
This is a <strong>beta estimate</strong>. Final pricing requires human review and may differ.
</div>
%s
<p style="font-size:13px;color:#666">Quantity: <strong>%d</strong></p>
<table style="width:100%%;border-collapse:collapse;font-size:14px">
<thead><tr style="border-bottom:2px solid #ddd">
<th style="text-align:left;padding:6px">File</th>
<th style="text-align:right;padding:6px">Entities</th>
<th style="text-align:right;padding:6px">Cost</th>
</tr></thead>
<tbody>%s</tbody>
<tfoot>
<tr style="border-top:1px solid #ddd"><td><strong>Cutting</strong></td><td></td><td style="text-align:right"><strong>$%.2f</strong></td></tr>
%s
<tr style="border-top:2px solid #ff4d2d"><td><strong style="font-size:16px">ESTIMATED TOTAL</strong></td><td></td><td style="text-align:right"><strong style="font-size:16px;color:#ff4d2d">$%.2f</strong></td></tr>
</tfoot>
</table>
<hr style="margin:20px 0;border:none;border-top:1px solid #eee">
<p style="font-size:12px;color:#999">American Laser Co. | americanlaserco@gmail.com<br>
Reply to this email to confirm your order or ask questions.</p>
</div>""" % (
            "<p style='font-size:13px;color:#666'>Material: <strong>%s</strong>%s</p>" % (material_name, " (customer provided)" if own_material else "") if material_name else "",
            qty, file_rows, cutting_total, mat_row, grand_total
        )

        SHOP_EMAIL = "americanlaserco@gmail.com"
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "American Laser Co. - Quote Estimate ($%.2f)" % grand_total
        msg["From"] = SHOP_EMAIL
        msg["To"] = "%s, %s" % (client_email, SHOP_EMAIL)
        msg["Reply-To"] = SHOP_EMAIL
        msg.attach(MIMEText(body_text, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        smtp_pass = os.environ.get("GMAIL_APP_PASSWORD", "")
        if not smtp_pass:
            log("Email: No GMAIL_APP_PASSWORD set, saving quote locally instead", "WARN")
            quote_dir = project_root / "quotes"
            quote_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            quote_path = quote_dir / ("quote_%s_%s.txt" % (ts, client_email.split("@")[0]))
            with open(quote_path, "w") as f:
                f.write(body_text)
            return jsonify({"success": True, "note": "Email saved locally (no SMTP credentials). Set GMAIL_APP_PASSWORD env var to enable sending.",
                           "saved_to": str(quote_path)})

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SHOP_EMAIL, smtp_pass)
            server.sendmail(SHOP_EMAIL, [client_email, SHOP_EMAIL], msg.as_string())

        log("Quote emailed to %s and %s ($%.2f)" % (client_email, SHOP_EMAIL, grand_total))
        return jsonify({"success": True})

    except Exception as e:
        log("Email send failed: %s" % e, "ERROR")
        return jsonify({"error": "Email send failed: %s" % e}), 500


# ---------------------------------------------------------------------------
# Bench-calibration endpoints (drives a bench Ruida via /api/bench-calibrate/*)
# Shop-local feature: requires a connected Ruida controller. Not used on
# the public website deployment.
# ---------------------------------------------------------------------------

from laser_time_tool import bench_calibrator as _bc

_bench_state = {"job": None, "connection": None, "mock": False}


@app.route("/api/bench-calibrate/discover", methods=["GET", "POST"])
def bench_discover():
    """Scan USB serial ports + local subnet for a connected Ruida.
    Optional ?mock=1 returns a mock connection for sandbox/UI testing."""
    if request.args.get("mock") == "1":
        info = {"kind": "mock", "card_id": "0xMOCK"}
        _bench_state["connection"] = info
        _bench_state["mock"] = True
        return jsonify({"found": True, "connection": info})
    info = _bc.discover_controller()
    if info is None:
        _bench_state["connection"] = None
        return jsonify({"found": False, "connection": None,
                        "hint": "Plug the bench Ruida into USB or set it on the LAN, then click Discover again."})
    _bench_state["connection"] = info
    _bench_state["mock"] = False
    return jsonify({"found": True, "connection": info})


@app.route("/api/bench-calibrate/start", methods=["POST"])
def bench_start():
    """Start a calibration run. Body JSON:
       {folders: [...], speeds: [11,16,20,100], force: false}"""
    if _bench_state["connection"] is None:
        return jsonify({"started": False, "error": "No controller connection. Run discover first."}), 400
    if _bench_state["job"] is not None and _bench_state["job"].status == "running":
        return jsonify({"started": False, "error": "A calibration run is already in progress."}), 400

    data = request.get_json(silent=True) or {}
    folders = data.get("folders") or [
        str(project_root / "test files"),
        str(project_root.parent / "convertedfiles"),
        str(project_root.parent / "final test files copy"),
    ]
    speeds = data.get("speeds") or [11.0, 16.0, 20.0, 100.0]
    force = bool(data.get("force", False))

    results_path = project_root / "bench_results.json"
    job = _bc.CalibrationJob(
        results_path=results_path,
        folders=folders,
        speeds=speeds,
        connection_info=_bench_state["connection"],
        force=force,
    )
    if not job.start():
        return jsonify({"started": False, "error": "Failed to start job thread."}), 500
    _bench_state["job"] = job
    return jsonify({"started": True})


@app.route("/api/bench-calibrate/status", methods=["GET"])
def bench_status():
    """Live status — frontend polls this every second while a job runs."""
    job = _bench_state["job"]
    if job is None:
        return jsonify({"status": "idle", "connection": _bench_state["connection"]})
    snap = job.snapshot()
    return jsonify(snap)


@app.route("/api/bench-calibrate/stop", methods=["POST"])
def bench_stop():
    job = _bench_state["job"]
    if job is None:
        return jsonify({"stopped": False, "reason": "no job running"})
    job.stop()
    return jsonify({"stopped": True, "status": job.snapshot()["status"]})


@app.route("/api/bench-calibrate/merge", methods=["POST"])
def bench_merge():
    """Fold controller-predicted times into calibration_table.json."""
    results_path = project_root / "bench_results.json"
    table_path = project_root / "laser_time_tool" / "calibration_table.json"
    if not results_path.exists():
        return jsonify({"merged": False, "error": "No bench_results.json yet."}), 400
    try:
        stats = _bc.merge_into_calibration(results_path, table_path)
        # Force reload of the in-memory calibration table on next use
        import laser_time_tool.calibration as _cal
        _cal._table = None
        return jsonify({"merged": True, **stats})
    except Exception as e:
        return jsonify({"merged": False, "error": str(e)}), 500


@app.route("/api/bench-calibrate/results", methods=["GET"])
def bench_results():
    """Return the full results list from disk."""
    results_path = project_root / "bench_results.json"
    if not results_path.exists():
        return jsonify({"results": []})
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        return jsonify({"results": data, "count": len(data)})
    except Exception as e:
        return jsonify({"results": [], "error": str(e)}), 500


@app.route("/bench-calibrate.html")
def bench_page():
    return send_from_directory(str(project_root), "bench-calibrate.html")


@app.route("/api/engine-quote", methods=["GET"])
def api_engine_quote():
    """Return the engine's CURRENT quote for a given stem + speed (DXF only),
    used by the bench-calibration UI to show side-by-side delta vs controller.
    """
    stem = request.args.get("stem", "").strip()
    try:
        speed = float(request.args.get("speed", "100"))
    except ValueError:
        return jsonify({"error": "bad speed"}), 400
    # Look for the source DXF in known folders
    search_dirs = [
        project_root / "test files",
        project_root.parent / "convertedfiles",
        project_root.parent / "final test files copy",
    ]
    dxf = None
    for d in search_dirs:
        cand = d / f"{stem}.dxf"
        if cand.exists():
            dxf = cand; break
    if dxf is None:
        return jsonify({"stem": stem, "speed": speed, "quote_seconds": None,
                       "reason": "no DXF found"})
    try:
        csm = {c: speed for c in range(1, 256)}
        cp = parse_dxf(str(dxf), csm)
        ents = sum(len(p) for p in cp.values())
        cut_mm = sum(path_length(pp) for paths in cp.valu