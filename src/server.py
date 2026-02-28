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
import numpy as np

from PIL import Image
import json
import os
import sys
import time
import logging
import threading
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
VIDEO_PLAYLIST = [
    os.path.join(PROJECT_ROOT, "data", "Forklift_Stops_Before_Obstruction.mp4"),
    os.path.join(PROJECT_ROOT, "data", "Forklift_Stops_Before_Chemical_Spill.mp4"),
    os.path.join(PROJECT_ROOT, "data", "Forklift_Dashcam_Footage_Generated.mp4")
]
INDEX_PATH = os.path.join(SRC_DIR, "index.html")

# ─── Initialize the Agent ───────────────────────────────────────────────────
agent = AutonomousForkliftAgent()
inference_lock = threading.Lock()


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
    Alternates through the VIDEO_PLAYLIST sequentially.
    """
    playlist_idx = 0
    while True:
        video_path = VIDEO_PLAYLIST[playlist_idx]
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.warning(f"[WARN] Could not open video: {video_path}")
            playlist_idx = (playlist_idx + 1) % len(VIDEO_PLAYLIST)
            time.sleep(1)
            continue

        logger.info(f"[STREAM] Playing: {os.path.basename(video_path)}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break # Video ended, move to next in playlist

            # Normalize to uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).clip(0, 255).astype(np.uint8) if frame.max() <= 1.0 else frame.clip(0, 255).astype(np.uint8)
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        
        cap.release()
        playlist_idx = (playlist_idx + 1) % len(VIDEO_PLAYLIST)


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
    Alternates through the VIDEO_PLAYLIST.
    """
    await websocket.accept()
    playlist_idx = 0
    frame_count = 0

    try:
        while True:
            video_path = VIDEO_PLAYLIST[playlist_idx]
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                await websocket.send_text(json.dumps({
                    "vision_analysis": f"ERROR: Could not load {os.path.basename(video_path)}",
                    "agent_action": {"tool": "maintain_course", "parameters": {"status": "io_error"}},
                }))
                playlist_idx = (playlist_idx + 1) % len(VIDEO_PLAYLIST)
                await asyncio.sleep(2)
                continue

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break # Next video

                # Normalize uint8
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8) if frame.max() <= 1.0 else frame.clip(0, 255).astype(np.uint8)

                frame_count += 1

                if frame_count % _FRAME_INTERVAL == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)

                    def locked_inference(img):
                        with inference_lock:
                            return agent.monitor_assess_act(img)

                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, locked_inference, pil_img)

                    # Augment result with current video source for UI visibility
                    result["video_source"] = os.path.basename(video_path)

                    # Logging
                    cycle = result.get("cycle_id", "?")
                    logger.info(f"━━━ Cycle #{cycle} [{result['video_source']}] ━━━━━━━━━━━━━━━━━━━━━━━━")
                    logger.info(f"  👁️  Vision: {result.get('vision_analysis', '')[:120]}")
                    
                    await websocket.send_text(json.dumps(result, default=str))

                await asyncio.sleep(0.03)
            
            cap.release()
            playlist_idx = (playlist_idx + 1) % len(VIDEO_PLAYLIST)

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception:
        import traceback
        logger.error(f"[WS] Exception:\n{traceback.format_exc()}")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  SWARMGRID-EDGE // Starting Edge Dashboard Server")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 60 + "\n")
    uvicorn.run(app, host=_HOST, port=_PORT, log_level="info")
