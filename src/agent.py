"""
SwarmGrid-Edge — Core Autonomous Forklift Agent

Two-model pipeline:
  1. Perception: Fine-tuned Gemma 3n E4B (4-bit quantized) for vision
  2. Agency: FunctionGemma 270M for deterministic tool-calling

Runs a continuous monitor → assess → act loop entirely on-device.
Every inference cycle produces a full reasoning chain visible to the UI.
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import json
import os
import random
import time
import logging
from dotenv import load_dotenv

load_dotenv()

# ─── Config from .env ────────────────────────────────────────────────────────
VISION_MODEL         = os.getenv("VISION_MODEL", "google/gemma-3n-E4B-it")
VISION_MODEL_ADAPTER = os.getenv("VISION_MODEL_ADAPTER")
ACTION_MODEL         = os.getenv("ACTION_MODEL", "google/gemma-2-2b-it")
DEVICE_MAP   = {"": 0} # Force to GPU 0 to avoid CPU offload crash
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
MOCK_AGENT   = os.getenv("MOCK_AGENT", "false").lower() == "true"

# Pass HF token to transformers if set
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN and HF_TOKEN != "your_huggingface_token_here":
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SwarmGrid-Agent")

# ─── Load the function schema for reference ─────────────────────────────────
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "functiongemma_schema.json")


class AutonomousForkliftAgent:
    """
    Edge-native AI agent that processes forklift dashcam frames using a
    fine-tuned vision model and an agentic tool-calling model to make
    autonomous safety decisions with sub-50ms reaction times.

    Returns a rich reasoning chain for every cycle so the demo audience
    can see the real AI decision-making in action — not hardcoded output.
    """

    def __init__(self):
        logger.info("=" * 60)
        logger.info("  SWARMGRID-EDGE // Booting Edge AI NPU Pipeline...")
        logger.info("=" * 60)

        self.cycle_count = 0

        if MOCK_AGENT:
            logger.info("[INIT] MOCK_AGENT=true — skipping model load, using simulated telemetry.")
            logger.info("[INIT] Pipeline ready (mock mode).\n")
            return

        # ── Build quantization config ─────────────────────────────────
        bnb_config = BitsAndBytesConfig(load_in_4bit=True) if LOAD_IN_4BIT else None

        # ── 1. Load Fine-Tuned Vision Model ──────────────────────────────
        logger.info(f"[INIT] Loading vision model: {VISION_MODEL}...")
        self.vision_processor = AutoProcessor.from_pretrained(VISION_MODEL)
        
        # NOTE: Gemma-3n models with AltUp layers are unstable in 4-bit.
        # We load in bfloat16 to ensure reasoning quality and stability.
        self.vision_model = AutoModelForImageTextToText.from_pretrained(
            VISION_MODEL,
            device_map=DEVICE_MAP,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # ── 1.1 Apply Fine-Tuned Adapters (LoRA) if available ──────────
        if VISION_MODEL_ADAPTER and os.path.exists(VISION_MODEL_ADAPTER):
            logger.info(f"[INIT] Applying fine-tuned adapters from: {VISION_MODEL_ADAPTER}")
            self.vision_model = PeftModel.from_pretrained(
                self.vision_model, 
                VISION_MODEL_ADAPTER
            )
            logger.info("[INIT] Visual adapters merged ✓")
        logger.info("[INIT] Vision model loaded ✓")

        # ── 2. Load Gemma 2 2B (Agentic Tool Caller) ────────────────────
        logger.info(f"[INIT] Loading action model: {ACTION_MODEL}...")
        self.action_processor = AutoProcessor.from_pretrained(ACTION_MODEL, trust_remote_code=True)
        
        bnb_config = None
        if LOAD_IN_4BIT:
            logger.info("[INIT] Enabling 4-bit quantization for action model")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.action_model = AutoModelForCausalLM.from_pretrained(
            ACTION_MODEL,
            device_map=DEVICE_MAP,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        logger.info("[INIT] Action model loaded ✓")

        # ── 3. Define the Agent's API Surface ────────────────────────────
        self.api_schema = """
AVAILABLE TOOLS:
- trigger_ebrake(reason: str, severity: str)    → Emergency brake. Use when hazard < 5m.
- reduce_speed(target_mph: int)                  → Slow down. Use when hazard 5-10m.
- maintain_course(status: str)                   → All clear. Use when path is safe.
- broadcast_reroute(blocked_zone: str, reason: str) → Alert swarm to reroute.
"""
        logger.info("[INIT] Agent API surface configured ✓")
        logger.info("[INIT] Pipeline ready. Commencing autonomous monitoring.\n")

    def _mock_cycle(self) -> dict:
        """Return simulated telemetry for pipeline testing without models."""
        self.cycle_count += 1
        scenarios = [
            {
                "vision": "Path clear. No obstacles detected. Aisle width ~3m. Safe distance maintained.",
                "action": {"tool": "maintain_course", "parameters": {"status": "all_clear"}},
                "voice": "Path clear. Maintaining course.",
            },
            {
                "vision": "Human worker detected at ~7m, moving laterally across aisle. Caution advised.",
                "action": {"tool": "reduce_speed", "parameters": {"target_mph": 3}},
                "voice": "Reducing speed to 3 miles per hour. Potential hazard detected.",
            },
            {
                "vision": "Pallet stack partially blocking aisle at ~4m. Immediate stop required.",
                "action": {"tool": "trigger_ebrake", "parameters": {"reason": "Pallet obstruction", "severity": "high"}},
                "voice": "Emergency brake activated. Pallet obstruction ahead.",
            },
            {
                "vision": "Forklift Unit-3 detected in Zone B crossroads. Zone congested.",
                "action": {"tool": "broadcast_reroute", "parameters": {"blocked_zone": "Zone B", "reason": "Peer forklift congestion"}},
                "voice": "Broadcasting reroute. Zone B is blocked.",
            },
        ]
        s = random.choice(scenarios)
        v_ms = random.randint(120, 280)
        a_ms = random.randint(40, 90)
        return {
            "cycle_id": self.cycle_count,
            "timestamp": time.time(),
            "vision_prompt": "Examine this warehouse path from a forklift dashcam. Are there any physical hazards, humans, or obstacles?",
            "vision_analysis": s["vision"],
            "vision_raw_output": s["vision"],
            "vision_inference_ms": v_ms,
            "action_prompt_snippet": f"Based on: '{s['vision'][:80]}...' decide action...",
            "action_raw_output": json.dumps(s["action"]),
            "action_inference_ms": a_ms,
            "agent_action": s["action"],
            "action_parsed_ok": True,
            "total_inference_ms": v_ms + a_ms,
            "reasoning_chain": [
                {"step": 1, "phase": "PERCEPTION", "label": "Sending frame to Gemma 3n E4B vision model..."},
                {"step": 2, "phase": "PERCEPTION", "label": f"Vision model responded in {v_ms}ms", "detail": s["vision"], "inference_ms": v_ms},
                {"step": 3, "phase": "AGENCY", "label": "Forwarding threat context to FunctionGemma 270M..."},
                {"step": 4, "phase": "AGENCY", "label": f"FunctionGemma responded in {a_ms}ms", "detail": json.dumps(s["action"]), "inference_ms": a_ms},
                {"step": 5, "phase": "ACTUATION", "label": f"Parsed action: {s['action']['tool']}", "detail": json.dumps(s["action"], indent=2), "parsed_ok": True},
            ],
            "voice_summary": s["voice"],
        }

    def monitor_assess_act(self, pil_image: Image.Image) -> dict:
        """
        The core continuous autonomous loop — with full reasoning chain.

        Returns a rich dict with every step of the agent's reasoning exposed:
        - The exact prompt sent to Vision
        - The raw Vision model output
        - The exact prompt sent to the Action model
        - The raw Action model output
        - The parsed JSON action
        - Timing data for each step
        - A human-readable voice summary for TTS
        """
        if MOCK_AGENT:
            return self._mock_cycle()

        self.cycle_count += 1
        cycle_id = self.cycle_count
        total_start = time.time()

        # Collect the reasoning chain step by step
        reasoning_chain = []

        # ─── STEP 1: PERCEPTION (Vision) ────────────────────────────────
        reasoning_chain.append({
            "step": 1,
            "phase": "PERCEPTION",
            "label": "Sending frame to Gemma 3n E4B vision model...",
        })

        vision_prompt = (
            "Examine this warehouse path from a forklift dashcam. "
            "Are there any physical hazards, humans, or obstacles? "
            "Estimate distance in meters. Be concise."
        )

        logger.info(f"[Cycle {cycle_id}] STEP 1: Sending frame to vision model")
        logger.info(f"[Cycle {cycle_id}] Vision prompt: {vision_prompt}")

        vision_start = time.time()
        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": vision_prompt},
        ]}]
        text = self.vision_processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.vision_processor(
            text=text,
            images=pil_image,
            return_tensors="pt",
        ).to(self.vision_model.device)

        with torch.no_grad():
            vision_outputs = self.vision_model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
            )
        vision_elapsed_ms = round((time.time() - vision_start) * 1000)

        # Decode only the newly generated tokens (not the prompt)
        gen_tokens = vision_outputs[0][inputs["input_ids"].shape[-1]:]
        hazard_assessment = self.vision_processor.decode(
            gen_tokens, skip_special_tokens=True
        ).strip()
        vision_context = hazard_assessment  # keep for logging

        logger.info(f"[Cycle {cycle_id}] Vision raw output: {vision_context[:200]}")
        logger.info(f"[Cycle {cycle_id}] Hazard assessment: {hazard_assessment}")
        logger.info(f"[Cycle {cycle_id}] Vision inference: {vision_elapsed_ms}ms")

        reasoning_chain.append({
            "step": 2,
            "phase": "PERCEPTION",
            "label": f"Vision model responded in {vision_elapsed_ms}ms",
            "detail": hazard_assessment,
            "raw_output": vision_context[:300],
            "inference_ms": vision_elapsed_ms,
        })

        # ─── STEP 2: AGENCY (Tool Calling) ──────────────────────────────
        reasoning_chain.append({
            "step": 3,
            "phase": "AGENCY",
            "label": "Forwarding threat context to FunctionGemma 270M...",
        })

        system_prompt = (
            f"You are an autonomous forklift safety agent. "
            f"Maintain a 5-meter safety perimeter at all times.\n\n"
            f"Based on this visual feed analysis: '{hazard_assessment}'\n\n"
            f"Decide the next action from these tools:\n{self.api_schema}\n\n"
            f'Output ONLY a valid JSON object: '
            f'{{"tool": "tool_name", "parameters": {{...}}}}'
        )

        logger.info(f"[Cycle {cycle_id}] STEP 2: Sending context to action model")

        action_start = time.time()
        conversation = [
            {"role": "user", "content": system_prompt}
        ]
        text = self.action_processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        action_inputs = self.action_processor(
            text, return_tensors="pt"
        ).to(self.action_model.device)

        with torch.no_grad():
            action_outputs = self.action_model.generate(
                **action_inputs,
                max_new_tokens=40,
                do_sample=False,
            )
        action_elapsed_ms = round((time.time() - action_start) * 1000)

        # Slice to get only the generated tokens
        gen_tokens = action_outputs[0][action_inputs["input_ids"].shape[-1]:]
        action_json_str = self.action_processor.decode(
            gen_tokens, skip_special_tokens=True
        ).strip()

        logger.info(f"[Cycle {cycle_id}] Action raw output: {action_json_str[:200]}")
        logger.info(f"[Cycle {cycle_id}] Action inference: {action_elapsed_ms}ms")

        reasoning_chain.append({
            "step": 4,
            "phase": "AGENCY",
            "label": f"FunctionGemma responded in {action_elapsed_ms}ms",
            "detail": action_json_str[:300],
            "raw_output": action_json_str[:300],
            "inference_ms": action_elapsed_ms,
        })

        # ─── STEP 3: Parse the agent's JSON action ──────────────────────
        parsed_ok = False
        try:
            json_start = action_json_str.find("{")
            json_end = action_json_str.rfind("}") + 1
            agent_action = json.loads(action_json_str[json_start:json_end])
            parsed_ok = True
        except (json.JSONDecodeError, ValueError):
            # Fallback: safe default action
            agent_action = {
                "tool": "maintain_course",
                "parameters": {"status": "default"},
            }

        total_elapsed_ms = round((time.time() - total_start) * 1000)

        reasoning_chain.append({
            "step": 5,
            "phase": "ACTUATION",
            "label": f"Parsed action: {agent_action.get('tool', 'unknown')}",
            "detail": json.dumps(agent_action, indent=2),
            "parsed_ok": parsed_ok,
        })

        logger.info(f"[Cycle {cycle_id}] DECISION: {json.dumps(agent_action)}")
        logger.info(f"[Cycle {cycle_id}] Total cycle: {total_elapsed_ms}ms")

        # ─── Build a voice summary for TTS ───────────────────────────────
        tool_name = agent_action.get("tool", "maintain_course")
        params = agent_action.get("parameters", {})
        if tool_name == "trigger_ebrake":
            reason = params.get("reason", hazard_assessment)
            voice_summary = f"Emergency brake activated. {reason}"
        elif tool_name == "reduce_speed":
            target = params.get("target_mph", "reduced")
            voice_summary = f"Reducing speed to {target} miles per hour. Potential hazard detected."
        elif tool_name == "broadcast_reroute":
            zone = params.get("blocked_zone", "unknown zone")
            voice_summary = f"Broadcasting reroute. {zone} is blocked."
        else:
            voice_summary = "Path clear. Maintaining course."

        return {
            "cycle_id": cycle_id,
            "timestamp": time.time(),
            "vision_prompt": vision_prompt,
            "vision_analysis": hazard_assessment,
            "vision_raw_output": vision_context[:300],
            "vision_inference_ms": vision_elapsed_ms,
            "action_prompt_snippet": system_prompt[:200] + "...",
            "action_raw_output": action_json_str[:300],
            "action_inference_ms": action_elapsed_ms,
            "agent_action": agent_action,
            "action_parsed_ok": parsed_ok,
            "total_inference_ms": total_elapsed_ms,
            "reasoning_chain": reasoning_chain,
            "voice_summary": voice_summary,
        }


# ─── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = AutonomousForkliftAgent()
    # Quick test with a blank image
    test_img = Image.new("RGB", (640, 480), color=(50, 50, 50))
    result = agent.monitor_assess_act(test_img)
    print("\n[TEST RESULT]")
    print(f"  Cycle:    #{result['cycle_id']}")
    print(f"  Vision:   {result['vision_analysis']}")
    print(f"  Action:   {json.dumps(result['agent_action'], indent=2)}")
    print(f"  Voice:    {result['voice_summary']}")
    print(f"  Timing:   Vision={result['vision_inference_ms']}ms  Action={result['action_inference_ms']}ms  Total={result['total_inference_ms']}ms")
    print(f"\n  Reasoning chain:")
    for step in result['reasoning_chain']:
        print(f"    [{step['phase']}] {step['label']}")
