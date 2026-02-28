"""
SwarmGrid-Edge — FastAPI Edge Dashboard Server

Serves as the forklift's onboard compute interface:
  - Serves the industrial dashboard UI (index.html)
  - Streams video frames via multipart MJPEG response
  - Runs the agent loop every ~15 frames and pushes rich telemetry via WebSocket
  - Provides a /health endpoint for edge device monitoring
  - Logs every inference cycle to the terminal for live demo visibility

Launch:  uvicorn src.server:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2
import asyncio

from PIL import Image
import json
import os
import sys
import time
import logging
from dotenv import load_dotenv

load_dotenv()

_HOST = os.getenv("HOST", "0.0.0.0")
_PORT = int(os.getenv("PORT", "8000"))
_FRAME_INTERVAL = int(os.getenv("AGENT_FRAME_INTERVAL", "15"))

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SwarmGrid-Server")

# Add project root to path so we can import agent
sys.path.insert(0, os.path.dirname(__file__))
from agent import AutonomousForkliftAgent

# ─── App Configuration ───────────────────────────────────────────────────────
app = FastAPI(
    title="SwarmGrid-Edge // Forklift OS",
    description="Autonomous Industrial Forklift Agent — Edge Dashboard",
    version="1.0.0",
)

# ─── Paths ───────────────────────────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
VIDEO_PATH = os.getenv("VIDEO_PATH") or os.path.join(PROJECT_ROOT, "data", "demo_dashcam.mp4")
if not os.path.isabs(VIDEO_PATH):
    VIDEO_PATH = os.path.join(PROJECT_ROOT, VIDEO_PATH)
INDEX_PATH = os.path.join(SRC_DIR, "index.html")

# ─── Initialize the Agent ───────────────────────────────────────────────────
agent = AutonomousForkliftAgent()


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the Edge Dashboard UI."""
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/health")
async def health_check():
    """Edge device health endpoint."""
    return {
        "status": "online",
        "agent": "SwarmGrid-Edge v1.0",
        "models": {
            "vision": "gemma-3n-e4b (INT4)",
            "action": "functiongemma-270m-it",
        },
        "edge_mode": True,
        "cloud_dependency": False,
    }


# ─── Video Streaming (MJPEG) ────────────────────────────────────────────────

def generate_video_frames():
    """
    Generator that yields MJPEG frames from the dashcam video.
    Loops the video continuously to simulate a live camera feed.
    """
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {VIDEO_PATH}")
        print("[WARN] Place a dashcam .mp4 file at data/demo_dashcam.mp4")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue

        # Ensure uint8 (some codecs output float frames)
        if frame.dtype != "uint8":
            import numpy as np
            frame = np.clip(frame, 0, 255).astype("uint8")
        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.get("/video_feed")
async def video_feed():
    """Stream the dashcam video as MJPEG for the UI <img> tag."""
    return StreamingResponse(
        generate_video_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ─── WebSocket: Agent Telemetry ─────────────────────────────────────────────

@app.websocket("/ws/agent_telemetry")
async def agent_telemetry(websocket: WebSocket):
    """
    Runs the autonomous monitor→assess→act loop and streams
    the agent's 'thoughts' and JSON tool calls to the UI in real-time.
    """
    await websocket.accept()
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0

    if not cap.isOpened():
        await websocket.send_text(json.dumps({
            "vision_analysis": "ERROR: No video source found at data/demo_dashcam.mp4",
            "agent_action": {"tool": "maintain_course", "parameters": {"status": "no_video"}},
        }))
        await websocket.close()
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue

            frame_count += 1

            # Run AI agent every N frames (configured via AGENT_FRAME_INTERVAL in .env)
            if frame_count % _FRAME_INTERVAL == 0:
                # Ensure uint8 before color conversion
                if frame.dtype != "uint8":
                    import numpy as np
                    frame = np.clip(frame, 0, 255).astype("uint8")
                # Convert OpenCV BGR → PIL RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                # Run inference in thread pool so async event loop stays alive
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, agent.monitor_assess_act, pil_img
                )

                # Log to server terminal so the demo audience sees live AI reasoning
                cycle = result.get("cycle_id", "?")
                tool = result.get("agent_action", {}).get("tool", "?")
                v_ms = result.get("vision_inference_ms", "?")
                a_ms = result.get("action_inference_ms", "?")
                t_ms = result.get("total_inference_ms", "?")
                logger.info(f"━━━ Cycle #{cycle} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                logger.info(f"  👁️  Vision ({v_ms}ms): {result.get('vision_analysis', '')[:120]}")
                logger.info(f"  🤖 Action ({a_ms}ms): {tool} → {json.dumps(result.get('agent_action', {}))}")
                logger.info(f"  🔊 Voice: {result.get('voice_summary', '')}")
                logger.info(f"  ⏱️  Total: {t_ms}ms")

                # Send full telemetry payload to the UI (includes reasoning chain + voice)
                await websocket.send_text(json.dumps(result, default=str))

            # Simulate ~30fps video playback timing
            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        print("[WS] Client disconnected from telemetry stream.")
    except Exception as e:
        print(f"[WS] Connection closed: {e}")
    finally:
        cap.release()


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  SWARMGRID-EDGE // Starting Edge Dashboard Server")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 60 + "\n")
    uvicorn.run(app, host=_HOST, port=_PORT, log_level="info")
