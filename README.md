# SwarmGrid-Edge

On-device warehouse intelligence for industrial forklifts. Transforms standard dashcam feeds into autonomous perception nodes that reason and act — no cloud required.

## What it does

SwarmGrid-Edge runs a continuous **monitor → assess → act** loop entirely on the forklift's local compute unit. A fine-tuned vision model analyzes dashcam frames, classifies the hazard level as a single token (`0`, `1`, or `2`), and a Python router instantly maps that token to the correct JSON tool call — triggering an emergency brake, reducing speed, or broadcasting a reroute to the swarm.

- **Sub-50ms reaction times** — no network round trip
- **Zero-cloud resilience** — stays live if Wi-Fi drops
- **Privacy-first** — video never leaves the warehouse floor

## Architecture

```
[ Dashcam / MP4 Feed ]
        │
        ▼ (every ~15 frames via OpenCV)
[ Gemma 3n E4B — Fine-tuned Vision Model ]
  Prompt: "Output ONE digit: 0=clear, 1=caution, 2=e-brake"
        │
        ▼ (single token output)
[ Python Token Router ]
  "2" → trigger_ebrake(reason, severity)
  "1" → reduce_speed(target_mph)
  "0" → maintain_course(status)
        │
        ▼
[ FastAPI Server ]
  - MJPEG stream → browser <img>
  - WebSocket telemetry → live dashboard
  - Inference cache (disk-backed JSON)
```

## Project structure

```
SwarmGrid/
├── src/
│   ├── agent.py          # Core autonomous agent (AutonomousForkliftAgent)
│   ├── server.py         # FastAPI server — video stream, WebSocket, REST API
│   └── index.html        # Industrial dashboard UI
├── models/
│   ├── finetuned_gemma_warehouse/   # LoRA adapters for Gemma 3n E4B
│   ├── finetune_gemma_vision.py     # Fine-tuning script (Unsloth + QLoRA)
│   └── functiongemma_schema.json    # Tool schema reference
├── data/
│   ├── *.mp4                        # Dashcam video files
│   └── inference_cache.json         # Cached inference results (auto-generated)
├── requirements.txt
└── .env                             # Config (see below)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

PyTorch with CUDA must be installed separately (the `requirements.txt` note explains this):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Configure `.env`

Create a `.env` file in the project root:

```env
# Required: HuggingFace token (for downloading Gemma models)
HF_TOKEN=your_huggingface_token_here

# Vision model — defaults to the base Gemma 3n E4B
VISION_MODEL=google/gemma-3n-E4B-it

# Optional: path to fine-tuned LoRA adapters
VISION_MODEL_ADAPTER=models/finetuned_gemma_warehouse

# Set to true to skip model loading and use simulated telemetry
MOCK_AGENT=false

# 4-bit quantization (saves VRAM, disable if unstable)
LOAD_IN_4BIT=true

# Run agent inference every N frames
AGENT_FRAME_INTERVAL=15

# Server bind address and port
HOST=0.0.0.0
PORT=8000
```

### 3. Add dashcam footage

Drop `.mp4` files into the `data/` directory. The server auto-detects them and populates the video selector in the UI.

### 4. Launch

```bash
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in a browser.

On first load (or when switching videos), the server pre-analyzes all frames and caches results to `data/inference_cache.json`. Subsequent loads are instant.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard UI |
| `/video_feed` | GET | MJPEG video stream |
| `/ws/agent_telemetry` | WS | Live agent reasoning + timing data |
| `/api/videos` | GET | List available `.mp4` files |
| `/api/select_video` | POST | Switch active video (`?filename=foo.mp4`) |
| `/api/health` | GET | Edge device health check |

## Agent actions

The agent can emit four tool calls:

| Tool | Trigger condition |
|---|---|
| `maintain_course` | Path clear, no hazards |
| `reduce_speed(target_mph)` | Hazard detected 5–10m ahead |
| `trigger_ebrake(reason, severity)` | Immediate hazard < 5m |
| `broadcast_reroute(blocked_zone, reason)` | Zone congested, alert swarm |

## Mock mode

Set `MOCK_AGENT=true` in `.env` to run without loading any models. The agent returns randomized but realistic telemetry, which is useful for testing the dashboard and WebSocket pipeline on a machine without a GPU.

## Fine-tuning

The vision model was fine-tuned on warehouse obstacle data (bootstrapped from Roboflow Universe) using Unsloth + QLoRA (INT4). The training script is at [models/finetune_gemma_vision.py](models/finetune_gemma_vision.py). Fine-tuned LoRA adapters are loaded at startup if `VISION_MODEL_ADAPTER` is set and the path exists.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.4 (recommended: 8GB+ VRAM for Gemma 3n E4B in bfloat16)
- CUDA toolkit installed
