"""In-process test of POST /api/blueprints/ocr (no network server needed).

Run:  python scripts/test_ocr.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.main import app


def main() -> int:
    uploads = sorted(Path("data/uploads").glob("*.jpg"))
    if not uploads:
        print("SKIP: no .jpg in data/uploads to test against")
        return 0
    img = uploads[0]

    client = TestClient(app)
    with img.open("rb") as f:
        r = client.post("/api/blueprints/ocr", files={"file": ("plan.jpg", f, "image/jpeg")})

    print(f"HTTP {r.status_code}  ({img.name})")
    if r.status_code != 200:
        print(r.text[:800])
        return 1

    j = r.json()
    print(f"image_size: {j['image_size']}  detections: {len(j['detections'])}")
    for d in j["detections"][:12]:
        print(f"  {d['confidence']:.2f}  {d['text']!r}")

    assert j["detections"], "OCR returned no text"
    assert all("box" in d and len(d["box"]) == 4 for d in j["detections"])

    # Rejection path: non-image must be 415.
    r2 = client.post(
        "/api/blueprints/ocr",
        files={"file": ("x.dxf", b"notanimage", "application/octet-stream")},
    )
    print(f"non-image -> {r2.status_code} {r2.json().get('detail', '')}")
    assert r2.status_code == 415, f"expected 415, got {r2.status_code}"

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
