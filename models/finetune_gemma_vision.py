"""
SwarmGrid-Edge — Fine-Tuning Script
====================================
Fine-tunes Gemma 3n E4B on the Roboflow "Warehouse Obstacle Detection" dataset
using Unsloth + QLoRA (INT4), converting bounding-box annotations into
natural-language hazard captions the vision model learns to produce.

Dataset : https://universe.roboflow.com/test-za-warehouse/warehouse-obstacle-detection
          ~9.9k warehouse images | 14 classes | CC BY 4.0
Model   : google/gemma-3n-e4b  (fine-tuned → models/finetuned_gemma_warehouse/)

Usage:
    python models/finetune_gemma_vision.py

Requires:  unsloth  trl  datasets  roboflow  (all in requirements.txt)
           ROBOFLOW_API_KEY and HF_TOKEN must be set in .env
"""

import os, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
HF_TOKEN       = os.getenv("HF_TOKEN")
RF_API_KEY     = os.getenv("ROBOFLOW_API_KEY")
RF_WORKSPACE   = os.getenv("ROBOFLOW_WORKSPACE",  "test-za-warehouse")
RF_PROJECT     = os.getenv("ROBOFLOW_PROJECT",    "warehouse-obstacle-detection")
RF_VERSION     = int(os.getenv("ROBOFLOW_VERSION", "1"))

BASE_MODEL     = os.getenv("VISION_MODEL", "google/gemma-3n-e4b")
OUTPUT_DIR     = Path(__file__).parent / "finetuned_gemma_warehouse"
DATA_DIR       = Path(__file__).parent.parent / "data" / "roboflow_dataset"

MAX_SEQ_LENGTH = 2048
LORA_RANK      = 16
TRAIN_STEPS    = 60           # quick hackathon fine-tune (~10 min on GPU)
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
LR             = 2e-4

# The exact question the agent asks during inference — training matches this
VISION_QUESTION = (
    "Examine this warehouse path from a forklift dashcam. "
    "Are there any physical hazards, humans, or obstacles? "
    "Estimate distance in meters. Be concise."
)

# ─── Class → hazard severity map ─────────────────────────────────────────────
# Maps detected class names to forklift-relevant hazard phrases + severity.
HAZARD_MAP = {
    "person":     ("Human",             "critical"),
    "worker":     ("Worker",            "critical"),
    "pedestrian": ("Pedestrian",        "critical"),
    "forklift":   ("Forklift",          "high"),
    "pallet":     ("Pallet",            "medium"),
    "box":        ("Cardboard box",     "medium"),
    "boxes":      ("Stacked boxes",     "medium"),
    "crate":      ("Crate",             "medium"),
    "rack":       ("Shelving rack",     "medium"),
    "backpack":   ("Abandoned backpack","medium"),
    "suitcase":   ("Suitcase",          "medium"),
    "luggage":    ("Luggage",           "medium"),
    "spill":      ("Chemical spill",    "critical"),
    "obstacle":   ("Obstacle",          "high"),
}


def estimate_distance(bbox_area_fraction: float) -> str:
    """
    Rough distance estimate based on how much of the frame the bbox occupies.
    Large (>15%) → ~2-3m   Medium (5-15%) → ~4-6m   Small (<5%) → ~7-10m
    """
    if bbox_area_fraction > 0.15:
        return f"{max(1.0, round(2 + (0.15 - bbox_area_fraction) * 10, 1))}m"
    elif bbox_area_fraction > 0.05:
        d = round(4 + (0.10 - bbox_area_fraction) * 40, 1)
        return f"{min(6.0, max(3.5, d))}m"
    else:
        return f"{round(7 + (0.05 - bbox_area_fraction) * 200, 1)}m"


def annotation_to_caption(annotations: list, img_w: int, img_h: int) -> str:
    """
    Converts COCO bounding-box annotations for one image into a natural-language
    hazard description that matches what the agent expects at inference time.
    """
    if not annotations:
        return "Path clear. No hazards or obstacles detected in the driving path."

    img_area   = img_w * img_h
    detections = []
    for ann in annotations:
        class_name = ann["class_name"].lower()
        x, y, w, h = ann["bbox"]          # COCO: [x_min, y_min, width, height]
        bbox_frac  = (w * h) / img_area
        distance   = estimate_distance(bbox_frac)
        label, severity = HAZARD_MAP.get(
            class_name, (class_name.replace("_", " ").title(), "medium")
        )
        center_x = x + w / 2
        position = ("left side" if center_x < img_w * 0.33
                    else "right side" if center_x > img_w * 0.66
                    else "center of path")
        detections.append({"label": label, "dist": distance,
                            "position": position, "severity": severity})

    SEV = {"critical": 0, "high": 1, "medium": 2}
    detections.sort(key=lambda d: (SEV.get(d["severity"], 3), d["dist"]))

    parts   = [f"{d['label']} at {d['dist']} ({d['position']})" for d in detections]
    caption = ". ".join(parts) + "."
    top_sev = detections[0]["severity"]
    prefix  = ("HAZARD DETECTED — " if top_sev == "critical"
               else "Caution — " if top_sev == "high"
               else "Obstacle present — ")
    return prefix + caption


# ─── Step 1: Download dataset from Roboflow ──────────────────────────────────
def download_dataset():
    from roboflow import Roboflow

    if not RF_API_KEY or RF_API_KEY == "your_roboflow_api_key_here":
        raise ValueError(
            "ROBOFLOW_API_KEY not set in .env\n"
            "Get your key at: https://app.roboflow.com/settings/api"
        )

    print(f"[DATASET] Connecting to Roboflow...")
    rf      = Roboflow(api_key=RF_API_KEY)
    project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
    version = project.version(RF_VERSION)

    print(f"[DATASET] Downloading {RF_PROJECT} v{RF_VERSION} (COCO format)...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    version.download("coco", location=str(DATA_DIR))
    print(f"[DATASET] Downloaded to: {DATA_DIR}")


# ─── Step 2: Parse COCO JSON → instruction-tuning pairs ──────────────────────
def build_training_pairs(split: str = "train") -> list:
    coco_json  = DATA_DIR / split / "_annotations.coco.json"
    images_dir = DATA_DIR / split

    if not coco_json.exists():
        raise FileNotFoundError(f"COCO annotations not found at {coco_json}")

    with open(coco_json) as f:
        coco = json.load(f)

    id_to_img   = {img["id"]: img for img in coco["images"]}
    id_to_cat   = {cat["id"]: cat["name"] for cat in coco["categories"]}
    img_to_anns = {}
    for ann in coco["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append({
            "class_name": id_to_cat[ann["category_id"]],
            "bbox":       ann["bbox"],
        })

    pairs = []
    for img_id, img_info in id_to_img.items():
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            continue
        anns   = img_to_anns.get(img_id, [])
        answer = annotation_to_caption(anns, img_info["width"], img_info["height"])
        pairs.append({"image_path": str(img_path),
                      "question":   VISION_QUESTION,
                      "answer":     answer})

    print(f"[DATASET] Built {len(pairs)} training pairs from '{split}' split")
    return pairs


# ─── Step 3: Fine-tune with Unsloth + QLoRA ──────────────────────────────────
def finetune(pairs: list):
    from unsloth import FastLanguageModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    from PIL import Image

    if HF_TOKEN:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

    print(f"[TRAIN] Loading {BASE_MODEL} with Unsloth (4-bit QLoRA)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,
        load_in_4bit   = True,
    )

    print(f"[TRAIN] Applying QLoRA adapters (rank={LORA_RANK})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r              = LORA_RANK,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha     = LORA_RANK,
        lora_dropout   = 0,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 42,
    )

    def convert_to_conversation(sample: dict) -> dict:
        img = Image.open(sample["image_path"]).convert("RGB")
        return {
            "messages": [
                {"role": "user",      "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text":  sample["question"]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text",  "text":  sample["answer"]},
                ]},
            ]
        }

    print("[TRAIN] Building HuggingFace dataset...")
    hf_dataset = Dataset.from_list(pairs).map(convert_to_conversation, num_proc=1)

    print(f"[TRAIN] Starting fine-tune ({TRAIN_STEPS} steps)...")
    FastLanguageModel.for_training(model)

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = hf_dataset,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        args = SFTConfig(
            output_dir                  = str(OUTPUT_DIR / "checkpoints"),
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_steps                = 5,
            max_steps                   = TRAIN_STEPS,
            learning_rate               = LR,
            fp16                        = True,
            logging_steps               = 5,
            save_steps                  = 30,
            optim                       = "adamw_8bit",
            seed                        = 42,
            report_to                   = "none",
            remove_unused_columns       = False,
            dataset_text_field          = "",
            dataset_kwargs              = {"skip_prepare_dataset": True},
        ),
    )
    trainer.train()
    print("[TRAIN] Fine-tuning complete ✓")

    print(f"[TRAIN] Saving merged model to {OUTPUT_DIR}...")
    model.save_pretrained_merged(str(OUTPUT_DIR), tokenizer, save_method="merged_4bit")
    print(f"[TRAIN] Done → {OUTPUT_DIR}")
    print(f"[TRAIN] Set VISION_MODEL={OUTPUT_DIR} in .env to use in the demo.")


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SwarmGrid-Edge — Warehouse Hazard Fine-Tuning")
    print("=" * 60)

    download_dataset()

    pairs = build_training_pairs("train")
    if not pairs:
        raise RuntimeError("No training pairs built — check dataset download.")

    print("\n[PREVIEW] First 3 training pairs:")
    for p in pairs[:3]:
        print(f"  IMG : {Path(p['image_path']).name}")
        print(f"  ANS : {p['answer']}\n")

    finetune(pairs)
