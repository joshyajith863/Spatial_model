"""
Spatial Copilot — FastAPI entrypoint.

Run locally on your LAN with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Then from any device on the same network:
    http://<your-machine-local-ip>:8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles

from app.api import blueprints, query

app = FastAPI(
    title="Spatial Copilot",
    description="Blueprint-to-3D spatial assistant — local network demo build.",
    version="0.1.0",
    # Disable the built-in docs routes; we register our own below so that
    # Swagger UI is served from local static assets instead of the CDN.
    docs_url=None,
    redoc_url=None,
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


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    # Serve Swagger UI from app/static/swagger-ui rather than the jsdelivr CDN,
    # so the docs page renders even on networks that block external CDNs.
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/swagger-ui.css",
        swagger_favicon_url="/static/swagger-ui/favicon-32x32.png",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
    )


@app.get("/docs/oauth2-redirect", include_in_schema=False)
async def swagger_ui_oauth2_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/health")
def health_check():
    return {"status": "ok"}
