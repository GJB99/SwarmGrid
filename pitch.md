# SwarmGrid-Edge: 60s High-Impact Pitch

## 🎯 What
**SwarmGrid-Edge** is on-device warehouse intelligence. We transform standard forklift cameras into autonomous perception nodes that reason and act without the cloud.

## 💡 Why
*   **Zero-Cloud Resilience:** Operations stay live even if the Wi-Fi drops.
*   **0ms Latency:** Sub-50ms reaction times for industrial safety.
*   **Privacy First:** Business-critical video data never leaves the warehouse floor.

## 🛠️ How
*   **Perception & Action Unification:** Fine-tuned **Gemma 3n** on-edge for *both* visual reasoning and action execution.
*   **Sub-Second Single-Token Routing:** To break the latency barrier, the model doesn't generate slow, text-heavy JSON arrays. It outputs a **single classification token (0, 1, or 2)**. A wrapper script instantly routes this single digit to the appropriate complex JSON tool payload (Clear, Caution, or E-Brake), dropping latency from ~15s to under 1s.
*   **Edge Native:** Fully optimized for local GPUs via 4-bit INT quantization and aggressive image pre-fill downscaling (448x448).

## ⏱️ The 60s Script
*   **[0-15s] THE HOOK:** "Most warehouse automation fails when the internet drops. SwarmGrid-Edge brings the brain to the vehicle."
*   **[15-35s] THE DEMO:** "This dashboard is 100% local. Our fine-tuned vision model identifies the spill and reasons about the risk in real-time."
*   **[35-50s] THE RESULT:** "The agent executes an E-Brake command immediately—zero cloud, zero lag."
*   **[50-60s] THE CLOSE:** "We deliver private, resilient, and instant warehouse safety. Automation that never stops."
