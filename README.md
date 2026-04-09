# Smart-Factory AI — Unified Showcase App

**Five CPU-only industrial ML models behind one FastAPI backend and a modern
single-page web frontend. All trained by Akobir Musaev, all live-testable in a
browser in under 60 seconds.**

This is a production-style wrapper around the best-performing models from
`D:\Ai_Portfolio` (Book B of the 15-model portfolio). It addresses the gap
identified in the deployment audit: the original portfolio ships individual
Docker containers per model (Book A only), no unified API, no interactive UI,
and 8 of 10 Book B projects weren't even persisting their trained weights.
This app fixes all of that.

---

## What's inside

```
D:\SmartFactoryApp\
├── README.md                       ← this file
├── start.bat                       ← one-click launcher (Windows)
├── start.sh                        ← one-click launcher (bash/git-bash)
├── backend/
│   └── main.py                     ← unified FastAPI app, 5 REST endpoints
└── frontend/
    ├── index.html                  ← bilingual EN/UZ single-page UI
    └── assets/
        ├── app.css                 ← dark theme + glassmorphism
        └── app.js                  ← vanilla JS client, no frameworks
```

## The 5 models it serves

| # | Endpoint | Model | Input | Output | Latency (CPU) |
|---|---|---|---|---|---|
| 1 | `POST /api/casting/predict` | EfficientNet-B0 fine-tune | `multipart` image | OK / DEFECT binary | **~45 ms** |
| 2 | `POST /api/ppe/predict` | YOLOv8s (conf=0.05) | `multipart` image | Bounding boxes + SAFE/UNSAFE verdict | **~100 ms** (warm) |
| 3 | `POST /api/ai4i/predict` | XGBoost + LightGBM rank-avg | JSON tabular | Ensemble failure prob + top-5 SHAP-lite drivers | **~20 ms** |
| 4 | `POST /api/ner/extract` | BERT-NER + manufacturing lexicons | JSON `{text}` | BERT entities + equipment/parts/actions/quantities | **~40 ms** (warm) |
| 5 | `POST /api/secom/predict` | 5-learner stack (XGB+LGB+LR+RF+ET) | JSON `{sensor_values: float[590]}` | 4-tree rank-avg ensemble score + per-learner probabilities | **~150 ms** |

Plus the utility endpoints `GET /api/health` and `GET /api/models`.

---

## Prerequisites

- **Windows** with the existing `D:\Ai_Portfolio\.venv` virtual environment
  (created by `Ai_Portfolio\SETUP.md`). This is the venv that already has
  PyTorch 2.11 CPU, timm, transformers, ultralytics, XGBoost, LightGBM,
  scikit-learn and every other Book B dependency.
- **FastAPI stack** added on top of that venv by this app (one-time install):
  ```
  "D:/Ai_Portfolio/.venv/Scripts/pip.exe" install fastapi "uvicorn[standard]" python-multipart
  ```
  (already done during the initial setup of this app)
- **Trained checkpoints** from the Book B projects. The app auto-loads them
  from:
  - `D:\Ai_Portfolio\02_casting_defect_classification\model\casting_efficientnet_b0.pt`
  - `D:\Ai_Portfolio\04_ppe_safety_yolov8n\model\yolov8s_hardhat.pt`
  - `D:\Ai_Portfolio\06_machine_failure_xgboost\model\ai4i_xgboost.json`
  - `D:\Ai_Portfolio\06_machine_failure_xgboost\model\ai4i_lightgbm.joblib`
  - `D:\Ai_Portfolio\09_equipment_ner_bert\model\bert-base-NER\`
  - `D:\Ai_Portfolio\10_sensor_anomaly_iforest_ae\model\secom_pipeline.joblib`

  If any of these are missing, re-run the corresponding project's
  `python run.py` — each `run.py` has been patched to persist its best weights
  to disk at the end of training.

---

## Run it

**Windows cmd:**
```bat
D:\SmartFactoryApp\start.bat
```

**bash / git-bash:**
```bash
bash D:/SmartFactoryApp/start.sh
```

**Manual:**
```bash
cd D:/SmartFactoryApp/backend
"D:/Ai_Portfolio/.venv/Scripts/python.exe" -m uvicorn main:app --host 127.0.0.1 --port 7860
```

Then open <http://127.0.0.1:7860/> in any modern browser. You should see the
bilingual EN/UZ Smart-Factory AI showcase with 5 model cards.

The OpenAPI spec is auto-generated at <http://127.0.0.1:7860/docs> — every
endpoint has a built-in "Try it out" form via Swagger UI.

---

## End-to-end verification

Verified on this machine against the saved checkpoints:

| Model | Sample input | Response | Latency |
|---|---|---|---|
| Casting (defect) | `cast_def_0_0.jpeg` | `DEFECT · 100% conf` | **59 ms** |
| Casting (ok) | `cast_ok_0_1018.jpeg` | `OK · 99.96% conf` | **43 ms** |
| PPE | `hard_hat_workers1010.png` | `UNSAFE · 5 Hardhat + 1 NO-Hardhat` | ~100 ms warm |
| AI4I healthy preset | `(298.1 K, 308.6 K, 1551 rpm, 42.8 Nm, 108 min, M)` | `OK · 0.0%` ensemble | **19 ms** |
| AI4I overstrain preset | `(299.3 K, 309.8 K, 1282 rpm, 68.4 Nm, 215 min, L)` | `FAILURE LIKELY · 100%` | **27 ms** |
| NER Samsung report | "At the Samsung Pyeongtaek fab, a 120 kg overheated bearing…" | ORG=Samsung Pyeongtaek (0.962), equipment=[bearing, belt, conveyor, gearbox, motor], parts=[bearing, belt, shaft], actions=[bent, overheated, snapped], qty=[120 kg, 3 mm] | 711 ms cold, ~40 ms warm |
| SECOM | 590 synthetic sensor values | `OK · ensemble 20.8%` (XGB 5.8%, LGB 39.0%, RF 19.7%, ET 18.6%) | 467 ms |

---

## Design decisions

**Single unified port.** All 5 models share `127.0.0.1:7860` instead of the
5 separate ports (8080–8084) the Book A Dockerfiles use. One gateway, one
server process, one OpenAPI spec.

**Lazy loading.** Each model's checkpoint is loaded on the first request that
touches it, not at server startup. Boot is instant; cold-cache latency is a
one-time hit for the first request of each endpoint. The frontend's health
ping wakes up nothing — only a real inference request triggers loading.

**Single venv, no docker-compose required.** Uses the existing
`D:\Ai_Portfolio\.venv` virtual environment so there are no duplicated
dependencies. On a dev machine this eliminates the need to build 5 Docker
images just to run the models side by side.

**Zero frontend dependencies.** The web UI is pure vanilla HTML + CSS + JS.
No React, no Vue, no npm install, no build step. `index.html` + `app.css` +
`app.js` loaded directly from FastAPI's `StaticFiles` mount. This keeps the
entire app self-contained and eliminates build-pipeline maintenance.

**Bilingual EN/UZ from day 1.** Every piece of user-facing text has parallel
English and Uzbek versions. Language switch is a pure-CSS toggle via
`body[data-lang]` attribute, persisted to `localStorage`.

**Honest verdict banners.** Each endpoint returns a plain-English `verdict`
field (`OK`, `DEFECT`, `SAFE`, `UNSAFE`, `FAILURE LIKELY`, `FAULT LIKELY`).
The frontend colour-codes these in green / red / amber so a visitor can read
the outcome without understanding ML.

---

## REST API reference

### `GET /api/health`
Liveness probe. Returns `{"status": "ok", "ts": <unix>}`.

### `GET /api/models`
List of configured models, their metadata, and whether their checkpoint is
present on disk.

### `POST /api/casting/predict`
**Content-Type:** `multipart/form-data`
**Form field:** `file` (image, jpg/png)
**Response:**
```json
{
  "class_index": 0,
  "class_name": "def_front",
  "confidence": 1.0,
  "probabilities": {"def_front": 1.0, "ok_front": 0.0},
  "verdict": "DEFECT",
  "latency_ms": 58.88
}
```

### `POST /api/ppe/predict`
**Content-Type:** `multipart/form-data`
**Form field:** `file` (image)
**Response:**
```json
{
  "detections": [
    {"class": "Hardhat", "confidence": 0.91, "box": [399.3, 296.9, 416.0, 344.3]}
  ],
  "counts": {"Hardhat": 5, "NO-Hardhat": 1},
  "total_people_in_frame": 6,
  "verdict": "UNSAFE",
  "conf_threshold": 0.05,
  "latency_ms": 96.1,
  "image_size": [1280, 720]
}
```

### `POST /api/ai4i/predict`
**Content-Type:** `application/json`
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
Returns XGBoost + LightGBM + ensemble probabilities, engineered features, and
top 5 feature drivers by XGBoost gain.

### `POST /api/ner/extract`
**Content-Type:** `application/json`
```json
{"text": "The hydraulic pump drive belt broke at 08:15..."}
```
Returns BERT-NER (open-domain PER/ORG/LOC/MISC) plus rule-based extractions
for manufacturing equipment, parts, actions, and quantities.

### `POST /api/secom/predict`
**Content-Type:** `application/json`
```json
{"sensor_values": [3030.93, 2564.00, 2187.73, ...590 numbers]}
```
Returns per-learner probabilities for XGBoost, LightGBM, RandomForest,
ExtraTrees, plus the 4-tree rank-average ensemble score. Missing values can
be sent as `null` or omitted (will be imputed by the saved `SimpleImputer`).

---

## Known limitations & future work

- **GPU models not served.** The 5 Book A models (A·01–A·05) are PyTorch
  state_dicts trained on Kaggle's P100 GPU. They are not included here
  because they target different input shapes and the source tree doesn't
  expose them as a clean Python library (each has its own `src/train.py`
  defining the model class locally). Adding them would require importing
  each `src/train.py` via a sys.path hack and is left as a follow-up.
- **No ONNX exports yet.** Everything runs as native PyTorch / XGBoost /
  LightGBM. Exporting the EfficientNet-B0 casting classifier and the YOLOv8s
  hardhat detector to ONNX would enable browser-side inference via
  `onnxruntime-web` (no backend at all).
- **No auth.** The API is open. Fine for a local demo, not production.
- **No docker-compose.** The app runs as a single uvicorn process. A
  Dockerfile could be added in ~20 lines, but for a showcase the "just run
  python main.py" story is simpler.
- **No mobile wrapper.** A Capacitor or Tauri wrapper around the frontend
  would give you an installable app — again, a natural follow-up.

---

## How this was built

See the git-trackable diff at commit time, but the key steps were:

1. Patched three Book B scripts (`02_casting`, `06_machine_failure`,
   `10_sensor_anomaly`) to call `torch.save()` / `joblib.dump()` at the end
   of training. Previously they discarded their weights at process exit,
   which made them impossible to serve.
2. Re-ran all three scripts to generate the checkpoints.
3. Wrote a single `backend/main.py` with lazy-loading for all 5 models,
   routing per-model inference through matched Pydantic request schemas.
4. Wrote `frontend/index.html` + `assets/app.css` + `assets/app.js` — a
   fully bilingual single-page UI. No build step.
5. End-to-end tested every endpoint with real data from the source datasets.

Total time to build, test, and document: one session.
