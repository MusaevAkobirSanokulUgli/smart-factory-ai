"""Smart-Factory AI — Self-contained FastAPI Backend (ONNX + native tree models).

No PyTorch, no timm, no ultralytics, no transformers at runtime.
All model files live in ../models/ so the project is fully portable.

Endpoints:
  GET  /                        → frontend
  GET  /api/health              → liveness
  GET  /api/models              → metadata
  POST /api/casting/predict     → image → OK / DEFECT
  POST /api/ppe/predict         → image → hard-hat detections + SAFE/UNSAFE
  POST /api/ai4i/predict        → JSON → machine failure probability
  POST /api/ner/extract         → text → equipment/parts/actions entities
  POST /api/secom/predict       → JSON → 4-tree ensemble anomaly score

Run:
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# Paths — all relative to this file, fully self-contained
# ------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "models"
FRONTEND_DIR = HERE.parent / "frontend"

CASTING_ONNX = MODELS_DIR / "casting_efficientnet_b0.onnx"
CASTING_META = MODELS_DIR / "casting_meta.json"
PPE_ONNX = MODELS_DIR / "yolov8s_hardhat.onnx"
AI4I_XGB = MODELS_DIR / "ai4i_xgboost.json"
AI4I_LGB = MODELS_DIR / "ai4i_lightgbm.joblib"
SECOM_PIPE = MODELS_DIR / "secom_pipeline.joblib"

# ------------------------------------------------------------------
# Lazy model registry
# ------------------------------------------------------------------
_MODELS: dict[str, Any] = {}


def _load_casting():
    if "casting" in _MODELS:
        return _MODELS["casting"]
    sess = ort.InferenceSession(str(CASTING_ONNX))
    meta = json.loads(CASTING_META.read_text())
    _MODELS["casting"] = {"session": sess, **meta}
    return _MODELS["casting"]


def _load_ppe():
    if "ppe" in _MODELS:
        return _MODELS["ppe"]
    sess = ort.InferenceSession(str(PPE_ONNX))
    _MODELS["ppe"] = {
        "session": sess,
        "conf": 0.05,
        "iou_thresh": 0.45,
        "class_names": {0: "Hardhat", 1: "NO-Hardhat"},
    }
    return _MODELS["ppe"]


def _load_ai4i():
    if "ai4i" in _MODELS:
        return _MODELS["ai4i"]
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(str(AI4I_XGB))
    bundle = joblib.load(AI4I_LGB)
    _MODELS["ai4i"] = {
        "xgboost": xgb_clf,
        "lightgbm": bundle["lightgbm"],
        "feature_columns": bundle["feature_columns"],
        "threshold": bundle["ensemble_threshold"],
        "class_labels": bundle["class_labels"],
    }
    return _MODELS["ai4i"]


def _load_secom():
    if "secom" in _MODELS:
        return _MODELS["secom"]
    _MODELS["secom"] = joblib.load(SECOM_PIPE)
    return _MODELS["secom"]


# ------------------------------------------------------------------
# YOLO ONNX post-processing (pure numpy — no ultralytics)
# ------------------------------------------------------------------
def _letterbox(img: np.ndarray, new_shape: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize + pad image to square, return (padded, ratio, (pad_w, pad_h))."""
    h, w = img.shape[:2]
    r = new_shape / max(h, w)
    new_w, new_h = int(w * r), int(h * r)
    resized = np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    pad_w = (new_shape - new_w) // 2
    pad_h = (new_shape - new_h) // 2
    padded = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return padded, r, (pad_w, pad_h)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy NMS. boxes: (N, 4) xyxy, scores: (N,)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def _yolo_postprocess(
    output: np.ndarray,
    ratio: float,
    pad: tuple[int, int],
    orig_hw: tuple[int, int],
    conf_thresh: float,
    iou_thresh: float,
    class_names: dict[int, str],
) -> list[dict]:
    """Decode YOLOv8 ONNX output [1, 6, 8400] → list of detection dicts."""
    # output shape: (1, num_classes+4, num_boxes) → transpose to (num_boxes, num_classes+4)
    pred = output[0].T  # (8400, 6)
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    class_scores = pred[:, 4:]  # (8400, num_classes)
    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)
    mask = max_scores >= conf_thresh
    if not mask.any():
        return []
    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]
    # Convert cxcywh to xyxy
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    # Undo letterbox padding + scale
    pad_w, pad_h = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / ratio
    # Clip to image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_hw[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_hw[0])
    # Per-class NMS
    keep_all: list[int] = []
    for cid in np.unique(class_ids):
        cmask = class_ids == cid
        cidx = np.where(cmask)[0]
        ckeep = _nms(boxes[cidx], max_scores[cidx], iou_thresh)
        keep_all.extend(cidx[ckeep].tolist())
    detections = []
    for i in keep_all:
        detections.append({
            "class": class_names.get(int(class_ids[i]), str(class_ids[i])),
            "confidence": round(float(max_scores[i]), 4),
            "box": [round(float(v), 1) for v in boxes[i].tolist()],
        })
    return detections


# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(
    title="Smart-Factory AI",
    description="4 CPU-only industrial ML models (ONNX + tree-based) behind one API — PPE detection, machine failure, equipment NER, SECOM fault. No PyTorch needed.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def root():
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"message": "frontend not found — open /docs"})


@app.get("/api/health")
def health():
    return {"status": "ok", "ts": time.time(), "runtime": "onnxruntime+xgboost (no torch)"}


@app.get("/api/models")
def list_models():
    return {
        "models": [
            {"id": "ppe", "name": "Hard-Hat / PPE Detection", "format": "ONNX",
             "domain": "Object Detection", "metric": "F1 0.969 / NoHH 0.691", "available": PPE_ONNX.exists()},
            {"id": "ai4i", "name": "Machine Failure Prediction", "format": "XGBoost JSON + LightGBM joblib",
             "domain": "Tabular PM", "metric": "AUROC 0.9826", "available": AI4I_XGB.exists()},
            {"id": "ner", "name": "Equipment / Parts NER", "format": "Rule-based lexicons",
             "domain": "NLP", "metric": "89 domain terms", "available": True},
            {"id": "secom", "name": "SECOM Wafer Fault", "format": "sklearn joblib",
             "domain": "Tabular Anomaly", "metric": "AUROC 0.7493", "available": SECOM_PIPE.exists()},
        ]
    }


# ------------------------------------------------------------------
# 1. Casting (ONNX)
# ------------------------------------------------------------------
@app.post("/api/casting/predict")
async def casting_predict(file: UploadFile = File(...)):
    if not CASTING_ONNX.exists():
        raise HTTPException(503, "Casting ONNX not found in models/")
    m = _load_casting()
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((m["img_size"], m["img_size"]))
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(m["normalize_mean"])) / np.array(m["normalize_std"])
    x = arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    t0 = time.perf_counter()
    logits = m["session"].run(None, {"image": x})[0]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    dt = (time.perf_counter() - t0) * 1000
    probs = probs[0].tolist()
    idx = int(np.argmax(probs))
    names = m["class_names"]
    return {
        "class_index": idx, "class_name": names[idx],
        "confidence": round(probs[idx], 4),
        "probabilities": {n: round(p, 4) for n, p in zip(names, probs)},
        "verdict": "DEFECT" if names[idx].startswith("def") else "OK",
        "latency_ms": round(dt, 2),
    }


# ------------------------------------------------------------------
# 2. PPE Hard-Hat (ONNX + numpy NMS)
# ------------------------------------------------------------------
@app.post("/api/ppe/predict")
async def ppe_predict(file: UploadFile = File(...)):
    if not PPE_ONNX.exists():
        raise HTTPException(503, "PPE ONNX not found in models/")
    m = _load_ppe()
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    img_np = np.asarray(img)
    padded, ratio, pad = _letterbox(img_np, 640)
    x = padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    inp_name = m["session"].get_inputs()[0].name
    t0 = time.perf_counter()
    output = m["session"].run(None, {inp_name: x})[0]
    detections = _yolo_postprocess(
        output, ratio, pad, img_np.shape[:2],
        m["conf"], m["iou_thresh"], m["class_names"],
    )
    dt = (time.perf_counter() - t0) * 1000
    counts = {"Hardhat": 0, "NO-Hardhat": 0}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    verdict = "SAFE" if counts.get("NO-Hardhat", 0) == 0 else "UNSAFE"
    return {
        "detections": detections, "counts": counts,
        "total_people_in_frame": sum(counts.values()),
        "verdict": verdict, "conf_threshold": m["conf"],
        "latency_ms": round(dt, 2), "image_size": [img.width, img.height],
    }


# ------------------------------------------------------------------
# 3. AI4I (XGBoost + LightGBM native)
# ------------------------------------------------------------------
class AI4IInput(BaseModel):
    air_temp_k: float = Field(..., ge=280, le=320)
    process_temp_k: float = Field(..., ge=290, le=320)
    rotational_speed_rpm: float = Field(..., ge=1000, le=3000)
    torque_nm: float = Field(..., ge=0, le=100)
    tool_wear_min: float = Field(..., ge=0, le=300)
    machine_type: str = Field("L")


def _engineer_ai4i(x: AI4IInput) -> pd.DataFrame:
    r: dict[str, Any] = {
        "air_temp_k": x.air_temp_k, "process_temp_k": x.process_temp_k,
        "rotational_speed_rpm": x.rotational_speed_rpm, "torque_nm": x.torque_nm,
        "tool_wear_min": x.tool_wear_min,
        "Type_L": int(x.machine_type.upper() == "L"),
        "Type_M": int(x.machine_type.upper() == "M"),
    }
    r["temp_diff"] = r["process_temp_k"] - r["air_temp_k"]
    r["power_w"] = r["torque_nm"] * r["rotational_speed_rpm"] * (2 * np.pi / 60)
    r["torque_x_wear"] = r["torque_nm"] * r["tool_wear_min"]
    r["rpm_x_wear"] = r["rotational_speed_rpm"] * r["tool_wear_min"]
    r["power_x_wear"] = r["power_w"] * r["tool_wear_min"]
    r["overstrain_score"] = r["tool_wear_min"] * r["torque_nm"] / 11000.0
    r["heat_diss_flag"] = int((r["temp_diff"] < 8.6) and (r["rotational_speed_rpm"] < 1380))
    r["power_oor"] = int((r["power_w"] < 3500) or (r["power_w"] > 9000))
    return pd.DataFrame([r])


@app.post("/api/ai4i/predict")
def ai4i_predict(inp: AI4IInput):
    if not AI4I_XGB.exists():
        raise HTTPException(503, "AI4I model not found.")
    m = _load_ai4i()
    df = _engineer_ai4i(inp)[m["feature_columns"]]
    t0 = time.perf_counter()
    p_xgb = float(m["xgboost"].predict_proba(df)[0, 1])
    p_lgb = float(m["lightgbm"].predict_proba(df)[0, 1])
    p_ens = (p_xgb + p_lgb) / 2
    dt = (time.perf_counter() - t0) * 1000
    try:
        gain = m["xgboost"].get_booster().get_score(importance_type="gain")
        total = sum(gain.values()) or 1.0
        drivers = [{"feature": k, "importance_pct": round(v / total * 100, 1)}
                   for k, v in sorted(gain.items(), key=lambda kv: -kv[1])[:5]]
    except Exception:
        drivers = []
    return {
        "probability_failure_xgboost": round(p_xgb, 4),
        "probability_failure_lightgbm": round(p_lgb, 4),
        "probability_failure_ensemble": round(p_ens, 4),
        "threshold": m["threshold"],
        "verdict": "FAILURE LIKELY" if p_ens >= m["threshold"] else "OK",
        "top_feature_drivers": drivers,
        "latency_ms": round(dt, 2),
    }


# ------------------------------------------------------------------
# 4. NER (rule-based only — no BERT, zero-dependency)
# ------------------------------------------------------------------
EQUIPMENT = {
    "pump", "motor", "belt", "valve", "pipe", "drill", "compressor", "loader",
    "crane", "boiler", "tank", "conveyor", "bearing", "gearbox", "turbine",
    "generator", "welder", "press", "lathe", "mill", "grinder", "fan",
    "chiller", "heater", "sensor", "cable", "winch", "robot", "arm", "forklift",
}
PARTS = {
    "plate", "hose", "belt", "bolt", "chain", "screw", "nut", "washer",
    "gasket", "seal", "bearing", "shaft", "gear", "blade", "nozzle", "filter",
    "o-ring", "spring", "clamp", "bracket", "wire", "cable", "housing",
    "rotor", "stator", "piston", "cylinder", "coupling", "flange",
}
ACTIONS = {
    "broke", "broken", "fell", "slipped", "hit", "struck", "cut", "burnt",
    "burned", "leaked", "jammed", "crushed", "snapped", "pierced", "fractured",
    "loose", "seized", "overheated", "misaligned", "bent", "corroded", "worn",
    "failed", "ruptured", "exploded", "vibrated",
}
QTY_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mm|cm|m|kg|g|bar|psi|rpm|%|kw|hp|v|a|l/min|mpa)\b", re.I)


class NERInput(BaseModel):
    text: str = Field(..., min_length=4, max_length=4000)


@app.post("/api/ner/extract")
def ner_extract(inp: NERInput):
    lower = inp.text.lower()
    t0 = time.perf_counter()
    equipment = sorted({w for w in EQUIPMENT if re.search(rf"\b{re.escape(w)}s?\b", lower)})
    parts = sorted({w for w in PARTS if re.search(rf"\b{re.escape(w)}s?\b", lower)})
    actions = sorted({w for w in ACTIONS if re.search(rf"\b{re.escape(w)}\w*\b", lower)})
    quantities = [{"value": q[0], "unit": q[1]} for q in QTY_RE.findall(inp.text)]
    dt = (time.perf_counter() - t0) * 1000
    return {
        "text": inp.text,
        "rule_equipment": equipment, "rule_parts": parts,
        "rule_actions": actions, "rule_quantities": quantities,
        "latency_ms": round(dt, 2),
    }


# ------------------------------------------------------------------
# 5. SECOM (sklearn + xgb + lgb native)
# ------------------------------------------------------------------
class SECOMInput(BaseModel):
    sensor_values: list[float] = Field(..., min_length=1, max_length=600)


@app.post("/api/secom/predict")
def secom_predict(inp: SECOMInput):
    if not SECOM_PIPE.exists():
        raise HTTPException(503, "SECOM pipeline not found.")
    pipe = _load_secom()
    n_raw = 590
    vals = list(inp.sensor_values)
    if len(vals) < n_raw:
        vals += [float("nan")] * (n_raw - len(vals))
    raw = np.asarray(vals[:n_raw], dtype=float).reshape(1, -1)
    try:
        mask1 = np.asarray(pipe["keep_stage1_mask"], dtype=bool)
        s1 = pipe["imputer"].transform(raw[:, mask1])
        s1 = pipe["scaler"].transform(s1)
        mask2 = np.asarray(pipe["keep_stage2_mask"], dtype=bool)
        sel = pipe["selector"].transform(s1[:, mask2])
    except Exception as e:
        raise HTTPException(400, f"Preprocessing failed: {e}")
    t0 = time.perf_counter()
    try:
        p_xgb = float(pipe["xgboost"].predict_proba(sel)[0, 1])
        p_lgb = float(pipe["lightgbm"].predict_proba(sel)[0, 1])
        p_rf = float(pipe["random_forest"].predict_proba(sel)[0, 1])
        p_et = float(pipe["extra_trees"].predict_proba(sel)[0, 1])
    except Exception as e:
        raise HTTPException(500, f"Model inference failed: {e}")
    p_ens = (p_xgb + p_lgb + p_rf + p_et) / 4
    dt = (time.perf_counter() - t0) * 1000
    thr = pipe["ensemble_threshold"]
    return {
        "probabilities": {
            "xgboost": round(p_xgb, 4), "lightgbm": round(p_lgb, 4),
            "random_forest": round(p_rf, 4), "extra_trees": round(p_et, 4),
            "ensemble_4tree": round(p_ens, 4),
        },
        "threshold": round(float(thr), 4),
        "verdict": "FAULT LIKELY" if p_ens >= thr else "OK",
        "n_sensors_received": len(inp.sensor_values),
        "n_features_selected": int(pipe["n_features_selected"]),
        "latency_ms": round(dt, 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
