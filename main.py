"""
FaceDoor — Vercel/Cloud FastAPI Server
=======================================
Endpoints:
  GET  /                      → Admin panel (static/index.html)
  GET  /mobile                → Mobile registration page (static/mobile_register.html)
  POST /api/mobile_register   → Accept 5 base64 photos → ArcFace → save to faces.json
  GET  /api/faces             → List registered people  (JSON)
  GET  /api/faces.json        → Download the full faces.json file
  POST /api/faces.json        → Upload / merge a faces.json (sync from kiosk)
  DELETE /api/faces/{name}    → Remove a person from the database
  GET  /api/status            → Health + model status

Deploy:
  • Local dev:   uvicorn main:app --reload --port 5050
  • Vercel:      requires Pro plan (3 GB lambda). See vercel.json.
  • Better alt:  Railway / Render / Fly.io free tier with a Dockerfile.

Storage:
  • Local:   data/faces.json
  • Vercel:  /tmp/faces.json  (ephemeral — persists per warm instance only)
             For persistence, swap DB_PATH for S3/Supabase/Firestore.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

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

# On Vercel /tmp is the only writable dir; locally use data/
if os.getenv("VERCEL"):
    DB_PATH = Path("/tmp/faces.json")
else:
    DB_PATH = BASE_DIR / "data" / "faces.json"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── ArcFace model (lazy, loaded in background) ─────────────────────────────────
_deepface_model = None
_model_ready    = False
_model_lock     = threading.Lock()

def _load_model_bg():
    global _deepface_model, _model_ready
    try:
        log.info("Loading ArcFace model …")
        from deepface import DeepFace
        # Warm up by representing a blank image
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        DeepFace.represent(dummy, model_name="ArcFace", enforce_detection=False)
        with _model_lock:
            _deepface_model = DeepFace
            _model_ready    = True
        log.info("ArcFace model ready ✓")
    except Exception as exc:
        log.warning("ArcFace load failed: %s", exc)

threading.Thread(target=_load_model_bg, daemon=True).start()

# ── faces.json helpers ─────────────────────────────────────────────────────────
_db_lock = threading.Lock()

def load_db() -> dict[str, Any]:
    if DB_PATH.exists():
        try:
            return json.loads(DB_PATH.read_text())
        except Exception:
            pass
    return {}

def save_db(db: dict[str, Any]):
    DB_PATH.write_text(json.dumps(db, indent=2))

# ── ArcFace helpers ────────────────────────────────────────────────────────────
def b64_to_bgr(b64: str) -> np.ndarray:
    """Decode a base64 JPEG/PNG string to a BGR numpy array."""
    raw   = base64.b64decode(b64.split(",")[-1])   # strip data-url prefix if present
    img   = Image.open(io.BytesIO(raw)).convert("RGB")
    arr   = np.array(img)
    return arr[:, :, ::-1]                          # RGB → BGR

def get_embedding(bgr: np.ndarray) -> list[float] | None:
    """Extract a 512-D ArcFace embedding. Returns None if model not ready or no face."""
    with _model_lock:
        model = _deepface_model
    if model is None:
        return None
    try:
        res = model.represent(bgr, model_name="ArcFace", enforce_detection=True)
        if res:
            return res[0]["embedding"]
    except Exception as exc:
        log.debug("No face detected: %s", exc)
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
    db = load_db()
    return {
        "ok":          True,
        "model_ready": _model_ready,
        "registered":  len(db),
    }

# ── API: mobile registration ───────────────────────────────────────────────────

@app.post("/api/mobile_register")
async def mobile_register(request: Request):
    """
    Accept JSON: { "name": str, "images": ["data:image/jpeg;base64,...", ...] }
    Extracts ArcFace embeddings, stores in faces.json.
    Returns { "success": bool, "samples": int, "name": str }
    """
    body: dict = await request.json()
    name:   str  = (body.get("name") or "").strip()
    images: list = body.get("images") or []

    if not name:
        raise HTTPException(status_code=422, detail="name is required")
    if not images:
        raise HTTPException(status_code=422, detail="at least one image is required")

    embeddings: list[list[float]] = []
    face_b64_sample: str | None   = None

    for b64 in images:
        try:
            bgr = b64_to_bgr(b64)
        except Exception:
            continue

        if face_b64_sample is None:
            # Store first image as thumbnail (scaled down)
            img_small = Image.fromarray(bgr[:, :, ::-1]).resize((80, 80))
            buf = io.BytesIO()
            img_small.save(buf, format="JPEG", quality=70)
            face_b64_sample = base64.b64encode(buf.getvalue()).decode()

        if not _model_ready:
            log.info("Model not ready — storing image only, no embedding yet")
            continue

        emb = get_embedding(bgr)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings and not face_b64_sample:
        return JSONResponse({"success": False, "error": "No valid images received"})

    with _db_lock:
        db = load_db()
        entry = db.get(name, {"embeddings": [], "face_b64": None, "samples": 0})
        entry["embeddings"].extend(embeddings)
        entry["face_b64"] = face_b64_sample or entry.get("face_b64")
        entry["samples"]  = len(entry["embeddings"])
        db[name] = entry
        save_db(db)

    log.info("Registered '%s' with %d embedding(s)", name, len(embeddings))
    return {
        "success": True,
        "name":    name,
        "samples": len(embeddings),
        "stored":  len(db[name]["embeddings"]),
        "model_ready": _model_ready,
    }

# ── API: list faces ────────────────────────────────────────────────────────────

@app.get("/api/faces")
async def list_faces():
    """Return a summary list of all registered people."""
    db = load_db()
    faces = [
        {
            "name":     name,
            "samples":  entry.get("samples", len(entry.get("embeddings", []))),
            "face_b64": entry.get("face_b64"),
        }
        for name, entry in db.items()
    ]
    return {"faces": faces, "total": len(faces)}

# ── API: download faces.json ───────────────────────────────────────────────────

@app.get("/api/faces.json")
async def download_faces_json():
    """
    Download the full faces.json database (including raw embeddings).
    The kiosk can pull this periodically to sync remotely registered faces.
    """
    db = load_db()
    return JSONResponse(
        content=db,
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
        db = load_db()
        merged = 0
        for name, entry in body.items():
            db[name] = entry
            merged += 1
        save_db(db)
    return {"success": True, "merged": merged, "total": len(db)}

# ── API: delete face ───────────────────────────────────────────────────────────

@app.delete("/api/faces/{name}")
async def delete_face(name: str):
    """Remove a person from the database."""
    with _db_lock:
        db = load_db()
        if name not in db:
            raise HTTPException(status_code=404, detail=f"'{name}' not found")
        del db[name]
        save_db(db)
    return {"success": True, "deleted": name, "remaining": len(db)}

# ── Dev entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5050))
    log.info("FaceDoor Cloud server starting on port %d …", port)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
