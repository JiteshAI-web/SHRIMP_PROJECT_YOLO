from ultralytics import YOLO
import cv2
from pathlib import Path

def detect_shrimp_with_label(model_path, source, save_dir='../runs/detect/predict_label'):
    """
    Detect shrimp with bounding boxes + SMALL class name text
    """
    # Load model
    model = YOLO(model_path)

    # Run detection
    results = model.predict(
        source=source,
        conf=0.25,
        iou=0.45,
        save=True,
        save_txt=True,
        save_conf=True,
        project=save_dir,
        name='exp',
        exist_ok=True,
        show_labels=True,     # ✅ SHOW class name
        show_conf=False,      # ❌ hide confidence
        line_width=2,
    )

    print("\n✅ Detection completed with class labels!")
    print(f"Results saved to: {save_dir}/exp")

    return results


def detect_and_display(model_path, image_path):
    """
    Detect and display immediately with SMALL 'shrimp' label
    """
    model = YOLO(model_path)
    results = model(image_path)

    for result in results:
        annotated_img = result.plot(
            labels=True,        # ✅ show class name
            conf=False,         # ❌ hide confidence
            line_width=2,
            font_size=0.4,      # ✅ SMALL text size
        )

        cv2.imshow("Shrimp Detection (Labeled)", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Console output
        print(f"\nDetections in {image_path}:")
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            print(f"  - Class: {result.names[cls]}, Confidence: {conf:.2f}, Box: {coords}")


if __name__ == "__main__":
    MODEL_PATH = "runs/runs/detect/train/weights/best.pt"

    # Detect on folder
    detect_shrimp_with_label(MODEL_PATH, "../test_images/")

    # Detect on single image (display)
    # detect_and_display(MODEL_PATH, "../test_images/shrimp1.jpg")
