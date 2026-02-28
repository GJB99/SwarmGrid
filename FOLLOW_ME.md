# SwarmGrid-Edge — Partner Setup Guide

Everything you need to get the demo running from a fresh clone.  
**Time to live demo: ~15 min** (excluding model download; Gemma 3n is 15 GB, pre-download overnight if needed).

---

## Prerequisites

| What | Where |
|------|-------|
| Python 3.11 | https://www.python.org/downloads/ — tick "Add to PATH" |
| `uv` (fast package installer) | `pip install uv` in any terminal |
| NVIDIA GPU (8 GB+ VRAM) | Needed for 4-bit inference. CPU works but is very slow. |
| Hugging Face account | https://huggingface.co/join |
| Git | https://git-scm.com |

---

## Step 1 — Clone & enter the project

```powershell
cd "C:\Users\<yourname>\wherever"
git clone <repo-url> SwarmGrid
cd SwarmGrid
```

---

## Step 2 — Create the Python environment

```powershell
python -m venv .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

> **Note:** `unsloth` (fine-tuning only) sometimes needs a CUDA-specific wheel.  
> If it fails during install, skip it — it is **not needed to run the live demo**.  
> Run: `uv pip install fastapi uvicorn opencv-python torch transformers accelerate bitsandbytes pillow python-multipart python-dotenv trl datasets roboflow`

---

## Step 3 — Set up your `.env` file

The `.env` file is **not committed to git** (it holds secrets). Create it manually:

```powershell
copy .env.example .env   # if Guus left a .env.example, otherwise:
notepad .env             # or any text editor
```

Fill in these values:

```dotenv
# ── Required ─────────────────────────────────────────────────────────────────

# Hugging Face token — get it at https://huggingface.co/settings/tokens
# Must have "Read" permission. Required to download gated Gemma models.
HF_TOKEN=hf_YOUR_TOKEN_HERE

# ── Models (leave as-is unless you change hardware) ──────────────────────────
VISION_MODEL=google/gemma-3n-e4b
ACTION_MODEL=google/gemma-2-270m-it
DEVICE_MAP=auto
LOAD_IN_4BIT=true

# ── Demo video ───────────────────────────────────────────────────────────────
# Path to any warehouse/forklift .mp4 you want to use
# Place the file in data/ and update this line:
VIDEO_PATH=data/demo_dashcam.mp4

# ── Server ───────────────────────────────────────────────────────────────────
HOST=0.0.0.0
PORT=8000
AGENT_FRAME_INTERVAL=15

# ── Roboflow (only needed if running the fine-tuning script) ─────────────────
# Get API key at https://app.roboflow.com/settings/api
ROBOFLOW_API_KEY=your_roboflow_api_key_here
ROBOFLOW_WORKSPACE=test-za-warehouse
ROBOFLOW_PROJECT=warehouse-obstacle-detection
ROBOFLOW_VERSION=1
```

### Accept Gemma model terms

The Gemma models are gated. You must accept the terms **once** per account:

1. Go to https://huggingface.co/google/gemma-3n-e4b and click **"Access repository"**
2. Go to https://huggingface.co/google/gemma-2-270m-it and do the same

Log in with your token so HF knows who you are:

```powershell
.venv\Scripts\activate
python -c "from huggingface_hub import login; login(token='hf_YOUR_TOKEN_HERE')"
```

---

## Step 4 — Add a demo video

The agent needs a video to process. Drop any warehouse/forklift dashcam `.mp4` into the `data/` folder and name it `demo_dashcam.mp4`.

- **Quick option:** download a free stock video from https://pixabay.com/videos/search/forklift/ and rename it.
- Make sure the filename matches `VIDEO_PATH` in your `.env`.

If you have no video, the server will still start — it just won't stream frames.

---

## Step 5 — Run the demo

```powershell
.venv\Scripts\activate
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

**First run will download the models (~16 GB total).** This is normal — go get a coffee.  
Once you see `Application startup complete`, open your browser:

```
http://localhost:8000
```

You should see the 3-column dashboard:
- **Left:** live video feed + inference timing stats
- **Center:** real-time reasoning chain (Vision → Risk → Action → Policy)
- **Right:** raw JSON telemetry log

The browser will **speak the agent's decisions aloud** via Web Speech API (togglable in the top bar).

---

## Step 6 — What you're looking at (for the pitch)

Each agent cycle (every ~15 frames):

1. A frame is grabbed from the video
2. **Gemma 3n E4B** (vision) describes what it sees: obstacles, humans, distances
3. **Gemma 2 270M** (action) reads the description and emits a JSON tool call:
   - `trigger_ebrake` → dashboard flashes red, voice says "EMERGENCY BRAKE"
   - `reduce_speed` → yellow caution
   - `maintain_course` → green / clear path
4. The full reasoning chain (5 steps, with ms timings) is streamed to the UI via WebSocket

**Everything runs locally. There is no cloud call. This is the point.**

---

## Live Demo Script (copy this)

**0:00 — Wi-Fi drop:**  
Turn off Wi-Fi on the laptop while the dashboard is running.  
_"If this forklift relied on the cloud, it would be blind right now. A half-second of cloud latency means a forklift travels 7 feet before braking. We run on the edge because latency kills."_

**0:30 — Fine-tuning:**  
Point to `models/finetune_gemma_vision.py`.  
_"Base models don't know industrial safety. We fine-tuned Gemma 3n E4B on 9,900 warehouse images from Roboflow using QLoRA at rank 16 — quantized to 4-bit to fit on edge hardware."_

**1:00 — Agentic loop:**  
Point to the reasoning chain panel.  
_"My hands are off the keyboard. Vision passes context to the action model. The action model fires a JSON tool call. You're watching a fully autonomous, offline safety loop — powered entirely by Google AI."_

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`
Make sure you're in the project root, not inside `src/`:
```powershell
cd "C:\Users\<yourname>\SwarmGrid"
uvicorn src.server:app
```

### `OutOfMemoryError` / CUDA OOM
The 4-bit model needs ~6 GB VRAM. If your GPU is smaller:
```dotenv
# In .env:
LOAD_IN_4BIT=true   # already on — make sure this line is present
DEVICE_MAP=auto
```
As a last resort, force CPU (slow but works):
```dotenv
DEVICE_MAP=cpu
LOAD_IN_4BIT=false
```

### `load_in_4bit` TypeError
This project uses `BitsAndBytesConfig` from `transformers`. If you see:
```
TypeError: __init__() got an unexpected keyword argument 'load_in_4bit'
```
Your transformers version is too old. Run:
```powershell
uv pip install --upgrade transformers
```

### Model download stuck / fails authentication
```powershell
python -c "from huggingface_hub import login; login(token='hf_YOUR_TOKEN')"
```
Then retry the server.

### Port 8000 already in use
```powershell
uvicorn src.server:app --port 8001
# Then open http://localhost:8001
```

---

## Optional: Run the fine-tuning script

This is for the "we actually trained the model" proof — not required for the live demo.

1. Add your Roboflow API key to `.env` (get it at https://app.roboflow.com/settings/api)
2. Run:
```powershell
.venv\Scripts\activate
python models/finetune_gemma_vision.py
```
3. It will download the ~9,900-image dataset, build instruction-tuning pairs, then run 60 QLoRA steps (~10 min on an A100, longer on consumer GPU).
4. The fine-tuned model saves to `models/finetuned_gemma_warehouse/`.
5. To use it in the demo, update `.env`:
```dotenv
VISION_MODEL=models/finetuned_gemma_warehouse
```

---

## File map (quick reference)

```
SwarmGrid/
├── src/
│   ├── agent.py          ← two-model pipeline (Vision → Action)
│   ├── server.py         ← FastAPI + WebSocket + MJPEG server
│   └── index.html        ← dashboard UI (TTS, reasoning chain, telemetry)
├── models/
│   ├── finetune_gemma_vision.py   ← QLoRA training script (Roboflow dataset)
│   └── functiongemma_schema.json  ← tool API schema for action model
├── data/
│   └── demo_dashcam.mp4  ← YOU PROVIDE THIS (not in git)
├── .env                  ← YOU CREATE THIS (not in git)
├── requirements.txt
├── README.md             ← full technical spec
└── FOLLOW_ME.md          ← this file
```
