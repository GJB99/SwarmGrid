import cv2
import os
import sys
from pathlib import Path

# Always resolve paths relative to the project root (one level up from src/)
PROJECT_ROOT = Path(__file__).parent.parent

def extract_frames(video_path: str, output_dir: str, interval: int = 5) -> int:
    """
    Extract every Nth frame from a video and save as JPEG.
    Returns the number of frames saved. Raises if video cannot be opened.
    """
    video_path  = str(PROJECT_ROOT / video_path) if not os.path.isabs(video_path) else video_path
    output_dir  = str(PROJECT_ROOT / output_dir) if not os.path.isabs(output_dir) else output_dir

    if not os.path.exists(video_path):
        print(f"[SKIP] Video not found: {video_path}")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] OpenCV could not open: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] {os.path.basename(video_path)}: {total_frames} frames @ {fps:.1f} fps — "
          f"saving every {interval}th frame → {output_dir}")

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"[DONE] Saved {saved_count} frames → {output_dir}")
    return saved_count


if __name__ == "__main__":
    clips = [
        ("data/Forklift_Dashcam_Footage_Generated.mp4",   "data/demo_frames/clip1"),
        ("data/Forklift_Stops_Before_Chemical_Spill.mp4", "data/demo_frames/clip2"),
        ("data/Forklift_Stops_Before_Obstruction.mp4",    "data/demo_frames/clip3"),
    ]

    total = 0
    for video, outdir in clips:
        total += extract_frames(video, outdir, interval=5)

    print(f"\n[SUMMARY] Extraction complete — {total} frames saved across {len(clips)} clips")
    if total == 0:
        print("[WARN] No frames were saved. Check that the .mp4 files are in data/")
        sys.exit(1)
