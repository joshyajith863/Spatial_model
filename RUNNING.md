# Running Spatial Copilot on your local network

No deployment needed — this runs on one machine and other devices (phone for
WebXR, laptop for the HUD) connect over Wi-Fi.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

`--host 0.0.0.0` is what makes it reachable from other devices on the LAN —
`127.0.0.1` would only work on the host machine itself.

## Find your machine's local IP

```bash
# macOS / Linux
ifconfig | grep "inet "
# Windows
ipconfig
```

Look for something like `192.168.x.x`.

## Access it

- From the host machine: `http://localhost:8000`
- From any other device on the same Wi-Fi: `http://<your-local-ip>:8000`
- Interactive API docs (auto-generated): `http://<your-local-ip>:8000/docs`

## Notes for the demo

- Make sure all presenting devices are on the **same Wi-Fi network** — this
  won't work across different networks or with mobile data.
- Some networks (corporate/campus Wi-Fi, guest networks) block device-to-device
  traffic. A phone hotspot or a simple home router is the most reliable setup
  for a live demo.
- No HTTPS is configured — some browser features (e.g. certain WebXR/camera
  APIs) may require a secure context. If you hit this, `mkcert` can generate a
  local trusted cert quickly, or test WebXR features over `localhost` first.
