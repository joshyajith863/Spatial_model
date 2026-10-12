"""
Spatial Copilot — FastAPI entrypoint.

Run locally on your LAN with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Then from any device on the same network:
    http://<your-machine-local-ip>:8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import blueprints, query

app = FastAPI(
    title="Spatial Copilot",
    description="Blueprint-to-3D spatial assistant — local network demo build.",
    version="0.1.0",
)

# Wide-open CORS is fine for a local-network demo (no auth, no public exposure).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(blueprints.router, prefix="/api/blueprints", tags=["blueprints"])
app.include_router(query.router, prefix="/api/query", tags=["query"])

# Serves generated GLB files / HUD assets to the WebXR frontend.
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/health")
def health_check():
    return {"status": "ok"}
