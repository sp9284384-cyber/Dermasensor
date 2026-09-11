"""
DermaSensor - Backend API
===========================
Flask backend serving:
    GET  /                  -> the frontend (index.html)
    GET  /api/health        -> health check
    POST /api/predict       -> {temperature, ph_level, moisture} -> risk class
                                + rule-based breakdown + AI-generated
                                natural-language explanation

Explainability design
----------------------
Two layers are combined so every prediction is auditable, not a black box:

  1. ML layer   -> RandomForestClassifier (model.joblib) gives the risk
                   class + class probabilities + global feature importances.

  2. Rule layer -> the exact clinical rule_engine() used to LABEL the
                   training data (imported from generate_dataset.py) is
                   re-run on the same input, so we can show which specific
                   thresholds were crossed (e.g. "pH 8.1 > 7.5 alkaline
                   threshold").

  3. Language layer -> a natural-language explanation is generated from
                   (1) + (2). If an Ollama server running a Llama 3 model
                   is reachable (OLLAMA_URL / OLLAMA_MODEL env vars), that
                   LLM is used to phrase the explanation in clinician-
                   friendly language, grounded strictly in the rule/ML
                   findings passed to it (no invented numbers - it only
                   restates the findings we computed). If no LLM is
                   configured/reachable, a deterministic template built
                   from the same findings is used instead, so the API
                   works fully offline out of the box.
"""

import os
import traceback

import joblib
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from generate_dataset import rule_engine

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model.joblib")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
USE_LLM = os.environ.get("DERMASENSOR_USE_LLM", "auto")  # "auto" | "on" | "off"

app = Flask(__name__, static_folder=APP_DIR, static_url_path="")
CORS(app)

_bundle = joblib.load(MODEL_PATH)
MODEL = _bundle["model"]
MODEL_FEATURES = _bundle["features"]
MODEL_CLASSES = _bundle["classes"]

RISK_COPY = {
    "Low": "Wound readings are within the expected healthy healing range.",
    "Medium": "Early warning signs detected — readings are drifting outside "
              "the normal healing range and should be watched closely.",
    "High": "Multiple readings are consistent with an active infection risk "
            "and clinical review is recommended.",
}

FACTOR_LABEL = {
    "temperature": "temperature",
    "ph_level": "pH level",
    "moisture": "moisture",
}


def validate_input(data):
    errors = []
    for field, lo, hi in [
        ("temperature", 25.0, 45.0),
        ("ph_level", 0.0, 14.0),
        ("moisture", 0.0, 100.0),
    ]:
        if field not in data:
            errors.append(f"Missing field: {field}")
            continue
        try:
            val = float(data[field])
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number")
            continue
        if not (lo <= val <= hi):
            errors.append(f"{field} out of plausible sensor range ({lo}-{hi})")
    return errors


def build_rule_explanation(temp, ph, moisture, points):
    """Human-readable breakdown of exactly which thresholds fired."""
    lines = []

    if points["temperature"] == 2:
        lines.append(f"Temperature {temp:.1f}\u00b0C is above 38.5\u00b0C \u2192 strong sign of local inflammation (+2 risk points).")
    elif points["temperature"] == 1:
        lines.append(f"Temperature {temp:.1f}\u00b0C is between 37.5\u00b0C and 38.5\u00b0C \u2192 mild inflammation (+1 risk point).")
    else:
        lines.append(f"Temperature {temp:.1f}\u00b0C is within the normal range (below 37.5\u00b0C, 0 risk points).")

    if points["ph_level"] == 2:
        lines.append(f"pH {ph:.1f} is above 8.0 \u2192 strong alkaline shift associated with bacterial bioburden (+2 risk points).")
    elif points["ph_level"] == 1 and ph > 7.5:
        lines.append(f"pH {ph:.1f} is between 7.5 and 8.0 \u2192 early alkaline shift (+1 risk point).")
    elif points["ph_level"] == 1:
        lines.append(f"pH {ph:.1f} is below 4.0 \u2192 abnormally acidic, also flagged (+1 risk point).")
    else:
        lines.append(f"pH {ph:.1f} is within the healthy healing range (4.0-6.0 ideal, 0 risk points).")

    if points["moisture"] == 2:
        lines.append(f"Moisture {moisture:.0f}% is above 85% \u2192 excessive exudate, maceration/infection risk (+2 risk points).")
    elif points["moisture"] == 1:
        lines.append(f"Moisture {moisture:.0f}% is between 65% and 85% \u2192 elevated exudate (+1 risk point).")
    else:
        lines.append(f"Moisture {moisture:.0f}% is within the normal moist-healing range (0 risk points).")

    return lines


def template_explanation(risk_class, rule_lines, top_feature, probabilities):
    prob_str = ", ".join(f"{c}: {p*100:.1f}%" for c, p in probabilities.items())
    return (
        f"Risk classification: {risk_class}. {RISK_COPY[risk_class]}\n\n"
        f"Why: {' '.join(rule_lines)}\n\n"
        f"The model's decision was most strongly driven by {FACTOR_LABEL[top_feature]}, "
        f"consistent with the rule-based breakdown above. "
        f"Model confidence by class \u2014 {prob_str}."
    )


def llm_explanation(risk_class, rule_lines, top_feature, probabilities, temp, ph, moisture):
    """Ask a local Llama 3 (via Ollama) to phrase the explanation in plain
    clinical language, strictly grounded in the computed findings below.
    Falls back to the deterministic template on any error or timeout."""
    prompt = (
        "You are a clinical assistant explaining a wound-infection risk "
        "assessment to a nurse. Use ONLY the facts given below, do not "
        "invent any numbers. Be concise (3-4 sentences), plain language.\n\n"
        f"Sensor readings: temperature={temp:.1f}C, pH={ph:.1f}, moisture={moisture:.0f}%.\n"
        f"Risk classification: {risk_class}.\n"
        f"Rule findings:\n- " + "\n- ".join(rule_lines) + "\n"
        f"Most influential factor: {FACTOR_LABEL[top_feature]}.\n"
        f"Model confidence: " + ", ".join(f"{c} {p*100:.0f}%" for c, p in probabilities.items()) + "\n\n"
        "Write the explanation now:"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=8,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        if text:
            return text, True
    except Exception:
        pass
    return template_explanation(risk_class, rule_lines, top_feature, probabilities), False


@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_classes": MODEL_CLASSES})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    errors = validate_input(data)
    if errors:
        return jsonify({"error": "validation_failed", "details": errors}), 400

    temp = float(data["temperature"])
    ph = float(data["ph_level"])
    moisture = float(data["moisture"])

    try:
        # ML prediction
        X = [[temp, ph, moisture]]
        pred_class = MODEL.predict(X)[0]
        proba = MODEL.predict_proba(X)[0]
        probabilities = {cls: float(p) for cls, p in zip(MODEL.classes_, proba)}

        importances = dict(zip(MODEL_FEATURES, getattr(MODEL, "feature_importances_", [0, 0, 0])))
        top_feature = max(importances, key=importances.get)

        # Rule-engine ground truth (for auditability / explainability)
        rule_class, rule_score, rule_points = rule_engine(temp, ph, moisture)
        rule_lines = build_rule_explanation(temp, ph, moisture, rule_points)

        use_llm = USE_LLM == "on" or (USE_LLM == "auto" and _ollama_reachable())
        if use_llm:
            explanation, llm_used = llm_explanation(
                pred_class, rule_lines, top_feature, probabilities, temp, ph, moisture
            )
        else:
            explanation = template_explanation(pred_class, rule_lines, top_feature, probabilities)
            llm_used = False

        return jsonify({
            "risk_class": pred_class,
            "probabilities": probabilities,
            "rule_engine": {
                "risk_class": rule_class,
                "risk_score": rule_score,
                "points": rule_points,
            },
            "top_feature": top_feature,
            "feature_importance": importances,
            "rule_breakdown": rule_lines,
            "explanation": explanation,
            "explanation_source": "llama3-ollama" if llm_used else "rule-based-template",
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": "prediction_failed", "details": str(exc)}), 500


_ollama_status_cache = {"reachable": None}


def _ollama_reachable():
    if _ollama_status_cache["reachable"] is not None:
        return _ollama_status_cache["reachable"]
    try:
        r = requests.get(OLLAMA_URL.replace("/api/generate", "/"), timeout=1)
        reachable = r.status_code < 500
    except Exception:
        reachable = False
    _ollama_status_cache["reachable"] = reachable
    return reachable


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
