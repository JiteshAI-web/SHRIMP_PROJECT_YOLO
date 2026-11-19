import math
from ultralytics import YOLO
import cv2
from pathlib import Path
from typing import Union
import numpy as np


# ===================== CALIBRATION =====================
# Real tray diameter in centimeters (physical constant)
TRAY_DIAMETER_CM = 70.0

# Fallback tray diameter in pixels (used only if automatic detection fails)
FALLBACK_TRAY_DIAMETER_PIXELS = 1200.0
# =======================================================


def _find_tray_diameter_pixels(img: np.ndarray) -> float:
    """
    Automatically estimate tray diameter in pixels from the image.

    Assumes the tray is roughly circular and prominent in the image.
    Uses Hough Circle Transform to detect the largest circle.
    Returns diameter in pixels if found, otherwise 0.0.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    h, w = gray.shape[:2]

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) / 4,
        param1=100,
        param2=30,
        minRadius=int(min(h, w) * 0.3),
        maxRadius=int(min(h, w) * 0.6),
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Take the largest detected circle as the tray
        largest = max(circles, key=lambda c: c[2])
        radius = largest[2]
        diameter = float(2 * radius)
        return diameter

    return 0.0


def _compute_length_px_from_box(xyxy):
    """
    Compute orientation-aware shrimp length in pixels from a bounding box [x1, y1, x2, y2].
    Uses major axis for horizontal/vertical shrimp and diagonal for diagonally oriented shrimp.
    """
    x1, y1, x2, y2 = xyxy
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    major = max(w, h)
    minor = min(w, h)
    if major > 0 and (minor / major) > 0.5:
        return math.hypot(w, h)
    return major


def count_and_measure_shrimp(
    model_path: Union[str, Path],
    source: Union[str, Path],
    save_dir: Union[str, Path] = "../runs/detect/count_length",
):
    """
    Run detection, count shrimps, and estimate their length (in centimeters).

    For each image:
      1. Automatically estimate tray diameter in pixels.
      2. Compute pixel_to_cm_ratio = TRAY_DIAMETER_CM / tray_diameter_pixels.
      3. Convert shrimp bounding-box length (pixels) to centimeters.

    If tray detection fails, falls back to a predefined pixel diameter.

    - Accepts a single image, video, or directory as source.
    - Saves annotated images with shrimp length text drawn near each box.
    - Prints per-image counts and lengths and a total shrimp count.
    """
    model_path = Path(model_path)
    source = str(source)
    save_dir = str(save_dir)

    # Load trained model
    model = YOLO(str(model_path))

    results = model.predict(
        source=source,
        conf=0.25,
        iou=0.45,
        save=False,          # we will save with custom overlays below
        save_txt=True,       # keep YOLO txt outputs if needed
        save_conf=True,
        project=save_dir,
        name="exp",
        exist_ok=True,
        show_labels=False,
        show_conf=False,
    )

    total_shrimp = 0

    # Ensure save directory exists
    out_dir = Path(save_dir) / "exp"
    out_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        # Original image path
        img_path = Path(result.path)
        img_name = img_path.name

        # Get image (numpy array) and a copy for drawing
        img = result.orig_img.copy()

        # --------- Automatic tray calibration per image ----------
        tray_diameter_px = _find_tray_diameter_pixels(img)
        if tray_diameter_px <= 0:
            tray_diameter_px = FALLBACK_TRAY_DIAMETER_PIXELS
            print(
                f"\n[WARN] Tray not detected automatically in {img_name}, "
                f"using fallback diameter {tray_diameter_px:.1f} px"
            )
        else:
            print(
                f"\n[INFO] Detected tray diameter in {img_name}: "
                f"{tray_diameter_px:.1f} px"
            )

        pixel_to_cm_ratio = TRAY_DIAMETER_CM / tray_diameter_px
        # --------------------------------------------------------

        shrimp_lengths_cm = []

        for box in result.boxes:
            coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, coords)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Compute length in pixels then convert to centimeters
            length_px = _compute_length_px_from_box(coords)
            length_cm = length_px * pixel_to_cm_ratio
            shrimp_lengths_cm.append(length_cm)

            # Put text (length in cm) above the box
            text = f"{length_cm:.2f} cm"
            cv2.putText(
                img,
                text,
                (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Save annotated image
        out_path = out_dir / img_name
        cv2.imwrite(str(out_path), img)

        # Per-image stats
        count = len(shrimp_lengths_cm)
        total_shrimp += count

        if count > 0:
            avg_length = sum(shrimp_lengths_cm) / count
        else:
            avg_length = 0.0

        print(f"\nImage: {img_name}")
        print(f"  Shrimp count: {count}")
        if count > 0:
            print("  Lengths (cm): " + ", ".join(f"{l:.2f}" for l in shrimp_lengths_cm))
            print(f"  Average length: {avg_length:.2f} cm")
        else:
            print("  No shrimps detected.")

    print("\n============================")
    print("Detection & Measurement Done")
    print("============================")
    print(f"Total shrimps across all images: {total_shrimp}")
    print(f"Annotated images saved to: {out_dir}")

    return total_shrimp


if __name__ == "__main__":
    # Default model path: best weights from training
    MODEL_PATH = "runs/runs/detect/train/weights/best.pt"

    # Example: run on all test images
    SOURCE = "../test_images"

    count_and_measure_shrimp(MODEL_PATH, SOURCE)


