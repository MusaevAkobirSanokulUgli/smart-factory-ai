# Smart-Factory AI — Unified Showcase App

**Four CPU-only industrial ML models behind one FastAPI backend and a modern
bilingual (EN/UZ) web frontend. ONNX + native tree-based inference — no PyTorch
needed at runtime.**

Models served:
1. **PPE Hard-Hat Detection** — YOLOv8s ONNX, safety-biased confidence sweep
2. **Machine Failure Prediction** — XGBoost + LightGBM ensemble, SHAP-lite drivers
3. **Equipment / Parts NER** — rule-based manufacturing lexicons, 2 ms/text
4. **SECOM Wafer Fault Detection** — 4-tree rank-average ensemble (XGB+LGB+RF+ET)

---

## What's inside

```
smart-factory-ai/
├── README.md
├── Dockerfile                      ← production Docker image
├── render.yaml                     ← Render.com one-click deploy
├── requirements.txt                ← onnxruntime, xgboost, lightgbm, sklearn (NO torch)
├── start.bat / start.sh            ← local launchers
├── backend/
│   └── main.py                     ← FastAPI, 4 inference endpoints + health + models
├── frontend/
│   ├── index.html                  ← bilingual EN/UZ single-page UI
│   └── assets/
│       ├── app.css                 ← dark theme + glassmorphism + responsive
│       └── app.js                  ← vanilla JS client, drag-drop, live inference
└── models/                         ← self-contained, all models shipped in repo
    ├── yolov8s_hardhat.onnx        ← 44.7 MB  (PPE detection)
    ├── ai4i_xgboost.json           ← 0.9 MB   (machine failure, XGBoost half)
    ├── ai4i_lightgbm.joblib        ← 2.3 MB   (machine failure, LightGBM half)
    ├── secom_pipeline.joblib       ← 13.2 MB  (full SECOM 4-tree pipeline)
    ├── casting_efficientnet_b0.onnx      ← (available but not exposed in UI)
    └── casting_efficientnet_b0.onnx.data
```

---

## The 4 models

| # | Endpoint | Model | Input | Key metric | Latency (CPU) |
|---|---|---|---|---|---|
| 1 | `POST /api/ppe/predict` | YOLOv8s ONNX | image upload | Hardhat F1 0.969, NO-Hardhat F1 0.691, mAP@0.5 0.579 | **~109 ms** |
| 2 | `POST /api/ai4i/predict` | XGBoost + LightGBM | JSON (6 fields) | Binary AUROC 0.9826, F1 0.904 | **~6 ms** |
| 3 | `POST /api/ner/extract` | Rule-based lexicons | JSON `{text}` | 30 equipment + 30 parts + 27 action terms | **~2 ms** |
| 4 | `POST /api/secom/predict` | 4-tree rank-avg (XGB+LGB+RF+ET) | JSON `{sensor_values[590]}` | Ensemble AUROC 0.7493 | **~120 ms** |

Plus `GET /api/health` and `GET /api/models`.

---

## Run locally

**Option 1 — pip:**
```bash
pip install -r requirements.txt
cd backend
uvicorn main:app --host 0.0.0.0 --port 7860
```

**Option 2 — Docker:**
```bash
docker build -t smart-factory-ai .
docker run -p 7860:7860 smart-factory-ai
```

Then open http://localhost:7860/ — bilingual EN/UZ showcase with 4 model cards.

OpenAPI docs at http://localhost:7860/docs (Swagger UI with "Try it out").

---

## Deploy (free tier)

### Render.com
`render.yaml` is included. Connect the GitHub repo on https://dashboard.render.com → auto-detected as Docker.

### Fly.io
```bash
fly launch --dockerfile Dockerfile --internal-port 7860
fly deploy
```

### Docker anywhere
```bash
docker build -t smart-factory-ai .
docker run -p 7860:7860 smart-factory-ai
```

---

## Runtime — no PyTorch needed

The backend uses **ONNX Runtime** for the YOLOv8s model and **native XGBoost /
LightGBM / scikit-learn** for the tree-based models. Total installed size is
~200 MB instead of ~3.5 GB with PyTorch + timm + ultralytics + transformers.

| Component | Size |
|---|---|
| onnxruntime | ~50 MB |
| xgboost | ~50 MB |
| lightgbm | ~10 MB |
| scikit-learn | ~30 MB |
| Model files | ~61 MB |
| **Total** | **~200 MB** |

This makes the Docker image small enough for any free-tier hosting.

---

## REST API reference

### `GET /api/health`
```json
{"status": "ok", "ts": 1775722007.17, "runtime": "onnxruntime+xgboost (no torch)"}
```

### `GET /api/models`
Returns metadata for all 4 models and whether each checkpoint is present.

### `POST /api/ppe/predict`
**Input:** `multipart/form-data`, field `file` (image jpg/png)
**Output:**
```json
{
  "detections": [
    {"class": "Hardhat", "confidence": 0.91, "box": [399.3, 296.9, 416.0, 344.3]}
  ],
  "counts": {"Hardhat": 5, "NO-Hardhat": 1},
  "verdict": "UNSAFE",
  "conf_threshold": 0.05,
  "latency_ms": 109.2
}
```
Pure-numpy YOLOv8 post-processing (letterbox + NMS) — no ultralytics dependency.

### `POST /api/ai4i/predict`
**Input:** `application/json`
```json
{
  "air_temp_k": 298.1,
  "process_temp_k": 308.6,
  "rotational_speed_rpm": 1551,
  "torque_nm": 42.8,
  "tool_wear_min": 108,
  "machine_type": "M"
}
```
Returns XGBoost + LightGBM + ensemble probabilities, threshold, verdict,
and top-5 feature importance drivers.

### `POST /api/ner/extract`
**Input:** `application/json`
```json
{"text": "The hydraulic pump drive belt broke. A 40 mm bolt hit the crane."}
```
Returns rule-based extractions: equipment, parts, actions, quantities.
89 domain terms across 3 lexicons.

### `POST /api/secom/predict`
**Input:** `application/json`
```json
{"sensor_values": [3030.93, 2564.00, 2187.73, ...up to 590 numbers]}
```
Returns per-learner probabilities (XGBoost, LightGBM, RandomForest, ExtraTrees)
plus the 4-tree rank-average ensemble score. Missing values padded with NaN and
imputed by the saved sklearn `SimpleImputer`.

---

## Design decisions

- **Self-contained models.** All model files live in `models/` inside the repo.
  No external path dependencies. Clone and run.
- **ONNX for vision, native for trees.** YOLOv8s is exported to ONNX with a
  pure-numpy NMS post-processor. Tree-based models (XGBoost, LightGBM, sklearn)
  stay in their native format — no conversion overhead, no version friction.
- **Lazy loading.** Models load on first request, not at boot. Server starts
  in under a second; first inference of each model has a one-time cold-cache hit.
- **Zero frontend deps.** Pure HTML + CSS + JS. No React, no npm, no build step.
- **Bilingual EN/UZ.** Language switch via `body[data-lang]` + `localStorage`.
  Every user-facing string has parallel English and Uzbek versions.
- **Honest verdicts.** Each endpoint returns a plain-English `verdict` field
  (`OK`, `SAFE`, `UNSAFE`, `FAILURE LIKELY`, `FAULT LIKELY`). The frontend
  colour-codes these in green / red so non-ML users can read the result.

---

## Built by

**Akobir Musaev** — senior full-stack + AI/ML engineer.

Part of a 15-model smart-factory AI portfolio targeting Korean predictive-maintenance
and industrial-automation roles.
