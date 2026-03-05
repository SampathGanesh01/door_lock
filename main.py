"""
FaceDoor — Vercel/Cloud FastAPI Server
=======================================
Endpoints:
  GET  /                      → Admin panel (static/index.html)
  GET  /mobile                → Mobile registration page (static/mobile_register.html)
  POST /api/mobile_register   → Accept N base64 photos → ArcFace → save embeddings to MongoDB
  GET  /api/faces             → List registered people  (JSON summary)
  GET  /api/faces.json        → Download full faces.json WITH embeddings (for kiosk sync)
  POST /api/faces.json        → Upload / merge a faces.json (sync from kiosk)
  DELETE /api/faces/{name}    → Remove a person from the database
  GET  /api/status            → Health + model status

Fix notes (v2):
  • enforce_detection=False  — mobile selfies with unusual angle/lighting still processed
  • align=True everywhere    — embeddings from server and kiosk are now comparable
  • 224×224 face crop saved  — enough detail for re-embedding on kiosk
  • All per-image embeddings stored and returned — kiosk uses real embeddings, not re-computed
  • COSINE_THRESHOLD unified to 0.45 both sides
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import time
import random
from pathlib import Path
from typing import Any
import pymongo
from config import MONGO_URI, DATABASE_NAME, COLLECTION_NAME

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger("facedoor")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
STATIC    = BASE_DIR / "static"

# ── MongoDB Client ─────────────────────────────────────────────────────────────
mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client[DATABASE_NAME]
users_collection    = db[COLLECTION_NAME]
insight_collection  = db["insight_users"]   # separate collection for InsightFace embeddings

def generate_unique_id() -> str:
    """Generate a unique 4-digit ID."""
    while True:
        uid = str(random.randint(1000, 9999))
        if not users_collection.find_one({"user_id": uid}):
            return uid


# ── ArcFace model (lazy, loaded ON DEMAND via /api/load_arcface) ───────────────
_deepface_model = None
_model_ready    = False
_model_loading  = False   # guard against double-loading
_model_lock     = threading.Lock()

def _load_model_bg():
    global _deepface_model, _model_ready, _model_loading
    try:
        log.info("Loading ArcFace model …")
        from deepface import DeepFace
        # Warm up by representing a blank image
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        DeepFace.represent(dummy, model_name="ArcFace", enforce_detection=False)
        with _model_lock:
            _deepface_model = DeepFace
            _model_ready    = True
            _model_loading  = False
        log.info("ArcFace model ready ✓")
    except Exception as exc:
        with _model_lock:
            _model_loading = False
        log.warning("ArcFace load failed: %s", exc)

# NOTE: ArcFace is NOT auto-loaded at startup.
# Trigger it manually via  POST /api/load_arcface  (or the admin panel button).

# ── InsightFace buffalo_s (lazy, background) ───────────────────────────────────
_insight_app   = None
_insight_ready = False
_insight_lock  = threading.Lock()

def _load_insight_bg():
    global _insight_app, _insight_ready
    try:
        log.info("Loading InsightFace buffalo_s …")
        from insightface.app import FaceAnalysis
        ia = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        ia.prepare(ctx_id=0, det_size=(320, 320))
        with _insight_lock:
            _insight_app   = ia
            _insight_ready = True
        log.info("InsightFace buffalo_s ready ✓")
    except Exception as exc:
        log.warning("InsightFace load failed: %s", exc)

threading.Thread(target=_load_insight_bg, daemon=True).start()

# ── DB helpers ─────────────────────────────────────────────────────────────────
_db_lock = threading.Lock()

def load_db() -> dict[str, Any]:
    """Helper to maintain dict format for /api/faces.json compatibility."""
    docs = users_collection.find()
    result = {}
    for doc in docs:
        result[doc["name"]] = doc
    return result


# ── ArcFace helpers ────────────────────────────────────────────────────────────
def b64_to_bgr(b64: str) -> np.ndarray:
    """Decode a base64 JPEG/PNG string to a BGR numpy array."""
    raw   = base64.b64decode(b64.split(",")[-1])   # strip data-url prefix if present
    img   = Image.open(io.BytesIO(raw)).convert("RGB")
    arr   = np.array(img)
    return arr[:, :, ::-1]                          # RGB → BGR

def get_embedding(bgr: np.ndarray) -> list[float] | None:
    """Extract a 512-D ArcFace embedding.

    Uses enforce_detection=False so that mobile selfies with unusual angles,
    lighting, or partial crops are still embedded rather than silently dropped.
    align=True is required — must match the kiosk-side alignment setting so
    cosine distances are meaningful.
    Returns None only if the model is not ready or a fatal error occurs.
    """
    with _model_lock:
        model = _deepface_model
    if model is None:
        log.warning("ArcFace model not loaded yet — skipping embedding")
        return None
    try:
        res = model.represent(
            bgr,
            model_name="ArcFace",
            enforce_detection=False,   # mobile selfies may not be perfectly framed
            detector_backend="retinaface", # matches kiosk for exact crop alignment
            align=True,                # MUST match kiosk; mismatched align → high cosine distance
        )
        if res and res[0].get("embedding"):
            return res[0]["embedding"]
        log.warning("ArcFace returned empty result")
    except Exception as exc:
        log.warning("Embedding extraction failed: %s", exc)
    return None

def cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="FaceDoor Cloud", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML / CSS / JS)
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

# ── Page routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def admin_panel():
    """Serve the laptop admin panel."""
    p = STATIC / "index.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Admin panel not found")
    return HTMLResponse(p.read_text())

@app.get("/mobile", response_class=HTMLResponse, include_in_schema=False)
async def mobile_page():
    """Serve the mobile auto-guided registration page."""
    p = STATIC / "mobile_register.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Mobile page not found")
    return HTMLResponse(p.read_text())

# ── API: status ────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    return {
        "ok":          True,
        "model_ready": _model_ready,
        "registered":  users_collection.count_documents({}),
    }

# ── API: mobile registration ───────────────────────────────────────────────────

@app.post("/api/mobile_register")
async def mobile_register(request: Request):
    """
    Accept JSON: { "name": str, "images": ["data:image/jpeg;base64,...", ...] }
    Extracts ArcFace embeddings, stores in faces.json.
    Returns { "success": bool, "samples": int, "name": str }
    """
    try:
        body: dict = await request.json()
    except Exception as e:
        log.exception("Failed to parse JSON body:")
        return JSONResponse({"success": False, "error": f"JSON parse error: {e}"}, status_code=400)

    name:   str  = (body.get("name") or "").strip()
    images: list = body.get("images") or []

    log.info("mobile_register invoked for user '%s' with %d image(s).", name, len(images))

    if not name:
        log.error("Validation failed: Name is empty.")
        raise HTTPException(status_code=422, detail="name is required")
    if not images:
        log.error("Validation failed: No images provided.")
        raise HTTPException(status_code=422, detail="at least one image is required")

    embeddings: list[list[float]] = []
    face_b64_sample: str | None   = None

    for idx, b64 in enumerate(images):
        try:
            bgr = b64_to_bgr(b64)
        except Exception as e:
            log.exception("Error decoding image %d from Base64:", idx)
            continue

        if face_b64_sample is None:
            try:
                # Save a 224×224 crop — large enough for ArcFace re-embedding on kiosk
                # (old 80×80 was too blurry to re-embed accurately)
                img_crop = Image.fromarray(bgr[:, :, ::-1]).resize((224, 224), Image.LANCZOS)
                buf = io.BytesIO()
                img_crop.save(buf, format="JPEG", quality=85)
                face_b64_sample = base64.b64encode(buf.getvalue()).decode()
                log.info("Saved 224×224 face crop for image %d", idx)
            except Exception as e:
                log.exception("Failed to create face crop from image %d:", idx)

        if not _model_ready:
            log.warning("Model not ready — image %d will have no embedding this request", idx)
            continue

        try:
            emb = get_embedding(bgr)
            if emb is not None:
                embeddings.append(emb)
                log.info("Extracted embedding for image %d (total so far: %d)", idx, len(embeddings))
            else:
                log.warning("No embedding for image %d — face may be too angled or obscured", idx)
        except Exception as e:
            log.exception("Exception during get_embedding for image %d:", idx)

    log.info("Embedding extraction complete: %d/%d images produced embeddings",
             len(embeddings), len(images))

    if not embeddings and not face_b64_sample:
        log.error("No valid images could be processed or decoded for user '%s'.", name)
        return JSONResponse({"success": False, "error": "No valid images received"})

    try:
        with _db_lock:
            entry = users_collection.find_one({"name": name})
            if not entry:
                entry = {
                    "name":       name,
                    "user_id":    generate_unique_id(),
                    "embeddings": [],
                    "face_b64":   None,
                    "samples":    0
                }
                log.info("Creating new DB entry for '%s' (uid=%s)", name, entry["user_id"])
            else:
                log.info("Appending embeddings to existing entry for '%s'", name)

            # Accumulate all per-image embeddings — more samples = better recognition
            entry["embeddings"].extend(embeddings)
            # Keep the best (first) face crop — update only if we don't already have one
            if face_b64_sample and not entry.get("face_b64"):
                entry["face_b64"] = face_b64_sample
            entry["samples"] = len(entry["embeddings"])

            users_collection.update_one(
                {"name": name},
                {"$set": {
                    "user_id":    entry["user_id"],
                    "embeddings": entry["embeddings"],
                    "face_b64":   entry["face_b64"],
                    "samples":    entry["samples"],
                }},
                upsert=True
            )
            log.info("Saved '%s' — total embeddings in DB: %d", name, entry["samples"])
    except Exception as e:
        log.exception("Database insertion failed for user '%s':", name)
        return JSONResponse({"success": False, "error": f"Database error: {e}"}, status_code=500)

    return {
        "success":    True,
        "name":       name,
        "user_id":    entry["user_id"],
        "samples":    len(embeddings),        # embeddings added this request
        "stored":     entry["samples"],        # total stored in DB
        "model_ready": _model_ready,
    }

# ── API: list faces ────────────────────────────────────────────────────────────

@app.get("/api/faces")
async def list_faces():
    """Return a summary list of all registered people."""
    docs = users_collection.find({}, {"_id": 0})
    faces = [
        {
            "name":     doc["name"],
            "user_id":  doc.get("user_id"),
            "samples":  doc.get("samples", len(doc.get("embeddings", []))),
            "face_b64": doc.get("face_b64"),
        }
        for doc in docs
    ]
    return {"faces": faces, "total": len(faces)}

# ── API: download faces.json ───────────────────────────────────────────────────

@app.get("/api/faces.json")
async def download_faces_json():
    """
    Download the full database WITH embeddings for kiosk sync.
    Format: { name: { user_id, embeddings: [[...512d...], ...], face_b64, samples } }

    IMPORTANT: This endpoint now includes real ArcFace embeddings so the kiosk
    does NOT need to re-generate embeddings from the (potentially small) thumbnail.
    This guarantees alignment consistency between server-registered and kiosk-detected faces.
    """
    docs = users_collection.find({}, {"_id": 0})
    result = {}
    for doc in docs:
        name = doc["name"]
        embs = doc.get("embeddings", [])
        result[name] = {
            "user_id":    doc.get("user_id"),
            "embeddings": embs,             # full 512-D vectors — kiosk uses these directly
            "face_b64":   doc.get("face_b64"),
            "samples":    len(embs),
        }
    return JSONResponse(
        content=result,
        headers={"Content-Disposition": "attachment; filename=faces.json"},
    )

# ── API: upload / merge faces.json ────────────────────────────────────────────

@app.post("/api/faces.json")
async def upload_faces_json(request: Request):
    """
    Upload a faces.json (from the local kiosk) to merge into the server DB.
    Existing entries are overwritten per name.
    """
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Expected a JSON object")
    with _db_lock:
        merged = 0
        for name, entry in body.items():
            if "user_id" not in entry:
                entry["user_id"] = generate_unique_id()
            users_collection.update_one(
                {"name": name},
                {"$set": entry},
                upsert=True
            )
            merged += 1
    total = users_collection.count_documents({})
    return {"success": True, "merged": merged, "total": total}

# ── API: delete face ───────────────────────────────────────────────────────────

@app.delete("/api/faces/{name}")
async def delete_face(name: str):
    """Remove a person from the database."""
    with _db_lock:
        result = users_collection.delete_one({"name": name})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"'{name}' not found")
    remaining = users_collection.count_documents({})
    return {"success": True, "deleted": name, "remaining": remaining}

# ── InsightFace routes ─────────────────────────────────────────────────────────

@app.post("/api/load_arcface")
async def load_arcface():
    """
    Manually trigger ArcFace model loading in the background.
    Safe to call multiple times — will not start a second thread if already loading or ready.
    """
    global _model_loading
    with _model_lock:
        if _model_ready:
            return {"status": "already_ready", "message": "ArcFace model is already loaded."}
        if _model_loading:
            return {"status": "loading", "message": "ArcFace model is already loading..."}
        _model_loading = True

    log.info("ArcFace load triggered via admin panel.")
    threading.Thread(target=_load_model_bg, daemon=True).start()
    return {"status": "started", "message": "ArcFace model loading started in background. Check status in ~30–60s."}


@app.get("/insight_register", response_class=HTMLResponse, include_in_schema=False)
async def insight_register_page():
    """Serve the single-photo kiosk registration page."""
    p = STATIC / "insight_mobile_register.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="insight_mobile_register.html not found")
    return HTMLResponse(p.read_text())


@app.get("/api/insight_status")
async def insight_status():
    return {
        "model_ready": _insight_ready,
        "registered":  insight_collection.count_documents({}),
    }


@app.post("/api/insight_register")
async def insight_register(request: Request):
    """
    Single-photo InsightFace registration.
    Body: { "name": str, "image": "data:image/jpeg;base64,..." }
    Returns: { "success": bool, "name": str, "user_id": str }
    """
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse({"success": False, "error": f"JSON parse error: {exc}"}, status_code=400)

    name = (body.get("name") or "").strip()
    b64  = (body.get("image") or "").strip()

    if not name:
        raise HTTPException(status_code=422, detail="name is required")
    if not b64:
        raise HTTPException(status_code=422, detail="image is required")
    if not _insight_ready:
        return JSONResponse(
            {"success": False, "error": "InsightFace model is still loading — retry in ~30s"},
            status_code=503,
        )

    # Decode base64 image → BGR numpy array (reuse existing b64_to_bgr helper)
    try:
        bgr = b64_to_bgr(b64)
    except Exception as exc:
        return JSONResponse({"success": False, "error": f"Image decode failed: {exc}"})

    # Extract InsightFace embedding
    with _insight_lock:
        faces = _insight_app.get(bgr)

    if not faces:
        return JSONResponse({
            "success": False,
            "error":   "No face detected — ensure good lighting and look directly at the camera",
        })

    embedding: list = faces[0].normed_embedding.tolist()   # 512-D normed

    # Build a 224×224 thumbnail for the admin panel
    face_b64: str | None = None
    try:
        img = Image.fromarray(bgr[:, :, ::-1]).resize((224, 224), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        face_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        log.warning("Thumbnail error: %s", exc)

    # Upsert in MongoDB
    with _db_lock:
        existing = insight_collection.find_one({"name": name})
        uid = existing["user_id"] if existing else generate_unique_id()
        insight_collection.update_one(
            {"name": name},
            {"$set": {
                "user_id":    uid,
                "embeddings": [embedding],
                "face_b64":   face_b64 or (existing or {}).get("face_b64"),
                "samples":    1,
            }},
            upsert=True,
        )
    log.info("InsightFace registered '%s' (uid=%s)", name, uid)
    return {"success": True, "name": name, "user_id": uid}


@app.get("/api/insight_faces.json")
async def download_insight_faces():
    """
    Download all InsightFace (buffalo_s) embeddings for kiosk sync.
    Format: { name: { user_id, embeddings: [[512-D]], face_b64, samples } }
    """
    docs = insight_collection.find({}, {"_id": 0})
    result = {
        doc["name"]: {
            "user_id":    doc.get("user_id"),
            "embeddings": doc.get("embeddings", []),
            "face_b64":   doc.get("face_b64"),
            "samples":    doc.get("samples", 0),
        }
        for doc in docs
    }
    return JSONResponse(result)


@app.delete("/api/insight_faces/{name}")
async def delete_insight_face(name: str):
    """Remove a person from the InsightFace database."""
    with _db_lock:
        result = insight_collection.delete_one({"name": name})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"'{name}' not found in insight_users")
    return {"success": True, "deleted": name}


# ── Dev entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5050))
    log.info("FaceDoor Cloud server starting on port %d …", port)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
