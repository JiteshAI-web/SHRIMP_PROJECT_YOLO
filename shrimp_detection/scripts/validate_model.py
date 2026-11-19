"""
Validate Trained YOLOv8 Shrimp Model (best.pt)
Runs evaluation metrics on validation dataset and performs sample inference.
"""

import sys
from pathlib import Path
import torch

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_PATH = SCRIPT_DIR / "runs" / "runs" / "detect" / "train" / "weights" / "best.pt"
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
TEST_IMG = PROJECT_ROOT / "test_images" / "1000199462.jpg"

def main():
    print("=" * 60)
    print("YOLOv8 Shrimp Detection - Model & Training Validation")
    print("=" * 60)

    # 1. Device check
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        device = "cpu"
        print("ℹ Running on CPU")

    # 2. Check model file
    if not MODEL_PATH.exists():
        print(f"❌ Model weights not found at: {MODEL_PATH}")
        return
    print(f"✅ Model weights: {MODEL_PATH} ({MODEL_PATH.stat().st_size / (1024*1024):.2f} MB)")

    # 3. Check data.yaml
    if not DATA_YAML.exists():
        print(f"❌ data.yaml not found at: {DATA_YAML}")
        return
    print(f"✅ Dataset config: {DATA_YAML}")

    # 4. Load model
    from ultralytics import YOLO
    print("\n📦 Loading model into memory...")
    model = YOLO(str(MODEL_PATH))

    # 5. Run validation on valid dataset
    print("\n📊 Running validation on valid dataset (27 images)...")
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=640,
        batch=16,
        device=device,
        verbose=False,
    )

    print("\n" + "=" * 60)
    print("🎯 VALIDATION RESULTS (best.pt)")
    print("=" * 60)
    print(f"  Precision:   {metrics.box.p[0]*100:.2f}%")
    print(f"  Recall:      {metrics.box.r[0]*100:.2f}%")
    print(f"  mAP@50:      {metrics.box.map50*100:.2f}%")
    print(f"  mAP@50-95:   {metrics.box.map*100:.2f}%")
    print(f"  Inference:   {metrics.speed.get('inference', 0):.2f} ms per image")
    print(f"  Preprocess:  {metrics.speed.get('preprocess', 0):.2f} ms")
    print(f"  Postprocess: {metrics.speed.get('postprocess', 0):.2f} ms")

    # 6. Run test inference
    if TEST_IMG.exists():
        print("\n" + "=" * 60)
        print("🧪 RUNNING INFERENCE ON SAMPLE TEST IMAGE")
        print("=" * 60)
        results = model.predict(source=str(TEST_IMG), conf=0.25, device=device, verbose=False)
        for r in results:
            boxes = r.boxes
            print(f"  Image: {TEST_IMG.name}")
            print(f"  Shrimps detected: {len(boxes)}")
            if len(boxes) > 0:
                confs = [round(float(c), 3) for c in boxes.conf]
                print(f"  Confidence scores: {confs[:8]}... (mean: {sum(confs)/len(confs):.2%})")

    print("\n" + "=" * 60)
    print("✅ TRAINING & MODEL VALIDATION SUCCESSFUL!")
    print("=" * 60)

if __name__ == "__main__":
    main()
