"""
Flask Web Application for Shrimp Detection System.
Displays real and AI-processed images from PostgreSQL database with indexed,
low-latency queries and smart Ollama AI aquaculture interpretation.
"""

import base64
import json
import math
import os
import random
import sys
import time
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from sqlalchemy import func
from sqlalchemy.orm import joinedload, sessionmaker

# Add parent directory to path for database imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))

from config import db_config
from llm_client import OllamaClient, generate_shrimp_interpretation
from models import AIImage, Base, ImageType, RealImage, get_engine
from prompts import get_prompt_template

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "shrimp-detection-secret-key-2024")

# Pooled Database Engine & Session Factory
engine = get_engine()
SessionFactory = sessionmaker(bind=engine)


def get_db_session():
    """Get database session."""
    return SessionFactory()


def image_to_base64(image_data, file_format):
    """Convert binary image data to base64 data URI."""
    if image_data is None:
        return None
    encoded = base64.b64encode(image_data).decode("utf-8")
    fmt = (file_format or "jpeg").lower()
    if fmt == "jpg":
        fmt = "jpeg"
    return f"data:image/{fmt};base64,{encoded}"


def extract_measurement_analytics(ai_img: AIImage) -> dict:
    """Compute size spread, uniformity (CV%), confidence, and biomass from measurements_json."""
    analytics = {
        "min_length_cm": None,
        "max_length_cm": None,
        "uniformity_pct": None,
        "avg_confidence_pct": None,
        "estimated_biomass_g": None,
    }

    if ai_img.shrimp_count and ai_img.avg_shrimp_weight_g:
        analytics["estimated_biomass_g"] = round(ai_img.shrimp_count * ai_img.avg_shrimp_weight_g, 2)

    if not ai_img.measurements_json:
        return analytics

    try:
        items = json.loads(ai_img.measurements_json)
        if not isinstance(items, list) or not items:
            return analytics

        lengths = [float(m["length_cm"]) for m in items if m.get("length_cm") is not None]
        confs = [float(m["confidence"]) for m in items if m.get("confidence") is not None]

        if lengths:
            mean_len = sum(lengths) / len(lengths)
            analytics["min_length_cm"] = round(min(lengths), 2)
            analytics["max_length_cm"] = round(max(lengths), 2)
            if mean_len > 0 and len(lengths) > 1:
                variance = sum((x - mean_len) ** 2 for x in lengths) / len(lengths)
                cv = (math.sqrt(variance) / mean_len) * 100.0
                uniformity = max(0.0, 100.0 - cv)
                analytics["uniformity_pct"] = f"{uniformity:.1f}%"
            else:
                analytics["uniformity_pct"] = "100.0%"

        if confs:
            analytics["avg_confidence_pct"] = f"{(sum(confs) / len(confs)) * 100.0:.1f}%"
    except (ValueError, TypeError, KeyError):
        pass

    return analytics


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/api/images")
def get_images():
    """
    API endpoint to get images for dashboard.
    Optimized to query only lightweight IDs and fetch binary blobs solely for the
    2 selected AI/real pairs, avoiding full-table BLOB scans.
    """
    session = get_db_session()

    try:
        # 1. Index-backed ID lookup (no binary image_data transferred)
        id_rows = (
            session.query(AIImage.id)
            .filter(AIImage.image_type == ImageType.COUNT_LENGTH)
            .all()
        )
        all_ids = [row[0] for row in id_rows]

        # Randomly select 2 AI image IDs
        num_to_show = min(2, len(all_ids))
        selected_ids = random.sample(all_ids, num_to_show) if all_ids else []

        # 2. Fetch ONLY the 2 selected AI images + their linked RealImages in 1 query
        selected_ai_images = []
        if selected_ids:
            rows = (
                session.query(AIImage)
                .options(joinedload(AIImage.real_image))
                .filter(AIImage.id.in_(selected_ids))
                .all()
            )
            by_id = {img.id: img for img in rows}
            selected_ai_images = [by_id[i] for i in selected_ids if i in by_id]

        data = {
            "real_images": [],
            "ai_images": [],
            "chart_data": [],
        }

        for ai_img in selected_ai_images:
            data["ai_images"].append({
                "id": ai_img.id,
                "name": ai_img.image_name,
                "data": image_to_base64(ai_img.image_data, ai_img.file_format),
                "shrimp_count": ai_img.shrimp_count,
                "avg_length_cm": ai_img.avg_shrimp_length_cm,
                "avg_weight_g": ai_img.avg_shrimp_weight_g,
                "pond_id": ai_img.pond_id,
                "cycle_no": ai_img.cycle_no,
                "device_id": ai_img.device_id,
                "water_depth_m": ai_img.water_depth_m,
                "created_at": ai_img.created_at.isoformat() if ai_img.created_at else None,
            })

            real_img = ai_img.real_image
            if real_img:
                data["real_images"].append({
                    "id": real_img.id,
                    "name": real_img.image_name,
                    "data": image_to_base64(real_img.image_data, real_img.file_format),
                    "created_at": real_img.created_at.isoformat() if real_img.created_at else None,
                })

        # 3. Chart data: lightweight 2-column indexed query (ZERO binary blobs fetched)
        chart_rows = (
            session.query(AIImage.shrimp_count, AIImage.created_at)
            .filter(AIImage.shrimp_count.isnot(None))
            .order_by(AIImage.created_at.asc())
            .limit(10)
            .all()
        )

        data["chart_data"] = [
            {
                "day": i + 1,
                "count": count or 0,
                "date": created_at.strftime("%Y-%m-%d") if created_at else "",
            }
            for i, (count, created_at) in enumerate(chart_rows)
        ]

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()


@app.route("/api/image/<int:image_id>")
def get_image(image_id):
    """Get a specific image by ID. Optional query param: ?type=real|ai"""
    session = get_db_session()
    img_kind = (request.args.get("type") or "").lower()

    try:
        img = None
        if img_kind == "ai":
            img = session.query(AIImage).filter_by(id=image_id).first()
        elif img_kind == "real":
            img = session.query(RealImage).filter_by(id=image_id).first()
        else:
            img = session.query(RealImage).filter_by(id=image_id).first()
            if not img:
                img = session.query(AIImage).filter_by(id=image_id).first()

        if img and img.image_data:
            fmt = (img.file_format or "jpeg").lower()
            if fmt == "jpg":
                fmt = "jpeg"
            response = send_file(BytesIO(img.image_data), mimetype=f"image/{fmt}")
            response.headers["Cache-Control"] = "public, max-age=3600"
            return response

        return jsonify({"error": "Image not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()


@app.route("/api/stats")
def get_stats():
    """Get database statistics using indexed GROUP BY queries."""
    session = get_db_session()

    try:
        real_rows = (
            session.query(RealImage.image_type, func.count(RealImage.id))
            .group_by(RealImage.image_type)
            .all()
        )
        ai_rows = (
            session.query(AIImage.image_type, func.count(AIImage.id))
            .group_by(AIImage.image_type)
            .all()
        )

        real_by_type = {t.value: c for t, c in real_rows if c > 0}
        ai_by_type = {t.value: c for t, c in ai_rows if c > 0}

        return jsonify({
            "total_real_images": sum(real_by_type.values()),
            "total_ai_images": sum(ai_by_type.values()),
            "real_by_type": real_by_type,
            "ai_by_type": ai_by_type,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()


@app.route("/api/latest-detection")
def get_latest_detection():
    """Get latest detection results without loading binary image data."""
    session = get_db_session()

    try:
        row = (
            session.query(
                AIImage.id,
                AIImage.shrimp_count,
                AIImage.avg_shrimp_length_cm,
                AIImage.avg_shrimp_weight_g,
                AIImage.model_used,
                AIImage.created_at,
            )
            .filter(AIImage.shrimp_count.isnot(None))
            .order_by(AIImage.created_at.desc())
            .first()
        )

        if row:
            return jsonify({
                "id": row.id,
                "shrimp_count": row.shrimp_count,
                "avg_length_cm": row.avg_shrimp_length_cm,
                "avg_weight_g": row.avg_shrimp_weight_g,
                "model_used": row.model_used,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            })

        return jsonify({})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()


@app.route("/api/health")
def health_check():
    """Lightweight health check for PostgreSQL latency and Ollama AI readiness."""
    session = get_db_session()
    try:
        t0 = time.perf_counter()
        ai_count = session.query(func.count(AIImage.id)).scalar()
        db_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        ollama = OllamaClient()
        ollama_ok = ollama.check_connection()
        model_name, supports_vision = ollama.resolve_model() if ollama_ok else (None, False)

        return jsonify({
            "status": "ok",
            "database": {
                "host": db_config.host,
                "name": db_config.database,
                "ai_images": ai_count,
                "query_ms": db_ms,
            },
            "ai_llm": {
                "ollama_connected": ollama_ok,
                "selected_model": model_name,
                "mode": ("vision" if supports_vision else "metrics") if model_name else None,
            },
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
    finally:
        session.close()


@app.route("/api/generate-interpretation", methods=["POST"])
def generate_interpretation():
    """Generate LLM-based shrimp interpretation for the displayed (or latest) AI image."""
    session = get_db_session()

    try:
        payload = request.get_json(silent=True) or {}
        template_type = payload.get("template_type", "main")
        ai_image_id = payload.get("ai_image_id")

        # Prefer the exact AI image currently shown in the dashboard
        target_ai = None
        if ai_image_id is not None:
            try:
                target_ai = (
                    session.query(AIImage)
                    .filter(AIImage.id == int(ai_image_id))
                    .first()
                )
            except (ValueError, TypeError):
                target_ai = None

        if target_ai is None:
            target_ai = (
                session.query(AIImage)
                .filter(
                    AIImage.image_type == ImageType.COUNT_LENGTH,
                    AIImage.shrimp_count.isnot(None),
                )
                .order_by(AIImage.created_at.desc())
                .first()
            )

        if not target_ai:
            return jsonify({"error": "No images available for analysis"}), 404

        # Compute detailed individual shrimp analytics from measurements_json
        analytics = extract_measurement_analytics(target_ai)

        metrics = {
            "ai_image_id": target_ai.id,
            "shrimp_count": target_ai.shrimp_count,
            "avg_length_cm": target_ai.avg_shrimp_length_cm,
            "avg_weight_g": target_ai.avg_shrimp_weight_g,
            "min_length_cm": analytics["min_length_cm"],
            "max_length_cm": analytics["max_length_cm"],
            "uniformity_pct": analytics["uniformity_pct"],
            "avg_confidence_pct": analytics["avg_confidence_pct"],
            "estimated_biomass_g": analytics["estimated_biomass_g"],
            "water_depth_m": target_ai.water_depth_m,
            "pond_id": target_ai.pond_id or "N/A",
            "cycle_no": target_ai.cycle_no or "N/A",
        }

        # Lightweight historical trend query (last 5 records, ZERO binary blobs fetched)
        last_5 = (
            session.query(AIImage.shrimp_count, AIImage.avg_shrimp_length_cm)
            .filter(AIImage.shrimp_count.isnot(None))
            .order_by(AIImage.created_at.desc())
            .limit(5)
            .all()
        )

        if len(last_5) > 1:
            count_trend = [row[0] for row in reversed(last_5)]
            length_trend = [round(row[1], 2) for row in reversed(last_5) if row[1] is not None]
            trend_desc = f"Shrimp count over last {len(count_trend)} measurements: {count_trend}. "
            if length_trend:
                trend_desc += f"Average length progression: {length_trend} cm."
            else:
                trend_desc += "Length data not available."
        else:
            trend_desc = "Insufficient historical data for trend analysis."

        # Raw base64 image (only sent if Ollama resolves a vision-capable model)
        image_base64 = (
            base64.b64encode(target_ai.image_data).decode("utf-8")
            if target_ai.image_data
            else None
        )

        vision_prompt = get_prompt_template(template_type)
        metrics_prompt = get_prompt_template("metrics" if template_type == "main" else template_type)

        llm_result = generate_shrimp_interpretation(
            image_base64=image_base64,
            metrics=metrics,
            history=trend_desc,
            prompt=vision_prompt,
            metrics_prompt=metrics_prompt,
        )

        if llm_result.get("interpretation"):
            return jsonify({
                "success": True,
                "interpretation": llm_result["interpretation"],
                "model_used": llm_result["model_used"],
                "mode": llm_result["mode"],
                "metrics": metrics,
                "trend": trend_desc,
            })

        return jsonify({
            "error": llm_result.get("error") or "Failed to generate interpretation. Ensure Ollama server is running.",
            "fallback": (
                f"Based on current data: Count={metrics['shrimp_count']}, "
                f"Length={metrics['avg_length_cm']} cm "
                f"(Range: {metrics['min_length_cm']}-{metrics['max_length_cm']} cm, "
                f"Uniformity: {metrics['uniformity_pct']}), "
                f"Weight={metrics['avg_weight_g']} g, "
                f"Est. Biomass={metrics['estimated_biomass_g']} g. "
                f"Trend: {trend_desc}"
            ),
        }), 503

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()


if __name__ == "__main__":
    print("Starting Shrimp Detection Web App...")
    print(f"Database: {db_config.database}")
    print(f"Host: {db_config.host}")

    # Create tables & indexes if they don't exist
    Base.metadata.create_all(engine)
    print("Database tables ready!")

    app.run(debug=True, host="0.0.0.0", port=5000)
