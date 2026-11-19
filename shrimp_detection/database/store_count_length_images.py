"""
Store Real Images and AI Bounding-Box Detection Images to PostgreSQL Database.

- Uses cross-platform paths relative to the shrimp_detection project root.
- If annotated AI images in runs/detect/count_length/exp are missing or are
  Git LFS pointer files, regenerates the exact bounding-box detection images
  from outputs/detection_results.json and test_images/ using OpenCV.
- Stores real images (test, train, validation) in real_images and linked
  AI bounding-box images + measurements/metadata in ai_images.
"""

import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
from tqdm import tqdm

# Add database directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_storage import ImageStorage
from models import AIImage, ImageType, RealImage, create_tables

# Cross-platform project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COUNT_LENGTH_DIR = PROJECT_ROOT / "runs" / "detect" / "count_length" / "exp"
TRAIN_DIR = PROJECT_ROOT / "data" / "train" / "images"
VALID_DIR = PROJECT_ROOT / "data" / "valid" / "images"
TEST_DIR = PROJECT_ROOT / "test_images"
DETECTION_RESULTS_JSON = PROJECT_ROOT / "outputs" / "detection_results.json"


def is_git_lfs_pointer(file_path: Path) -> bool:
    """Check if a file is a Git LFS pointer rather than actual binary image data."""
    if not file_path.exists() or file_path.stat().st_size > 1024:
        return False
    try:
        return file_path.read_bytes().startswith(b"version https://git-lfs.github.com")
    except OSError:
        return False


def load_detection_results() -> Dict[str, dict]:
    """Load precomputed YOLOv8 detection results indexed by image_name."""
    if not DETECTION_RESULTS_JSON.exists():
        return {}
    with open(DETECTION_RESULTS_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {rec["image_name"]: rec for rec in records if "image_name" in rec}


def compute_orientation_aware_length_px(bbox) -> float:
    """Compute orientation-aware shrimp length in pixels from [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    major = max(w, h)
    minor = min(w, h)
    if major > 0 and (minor / major) > 0.5:
        return math.hypot(w, h)
    return major


# Physical sanity bounds to reject false-positive huge/tiny bounding boxes (e.g. tray rim artifacts)
MIN_SHRIMP_LENGTH_CM = 0.5
MAX_SHRIMP_LENGTH_CM = 25.0


def recalc_detection_metrics(det: dict) -> Tuple[list, Optional[float]]:
    """
    Recompute orientation-aware shrimp lengths (cm) from stored bboxes and tray scale,
    filtering out physically implausible false-positive boxes (> 25 cm or < 0.5 cm).
    """
    ratio = det.get("pixel_to_cm_ratio")
    raw_measurements = det.get("measurements") or []
    updated = []
    lengths = []

    for m in raw_measurements:
        item = dict(m)
        bbox = item.get("bbox")
        if bbox and len(bbox) == 4 and ratio:
            length_cm = round(compute_orientation_aware_length_px(bbox) * float(ratio), 2)
            if not (MIN_SHRIMP_LENGTH_CM <= length_cm <= MAX_SHRIMP_LENGTH_CM):
                continue
            item["length_cm"] = length_cm
            lengths.append(length_cm)
        elif item.get("length_cm") is not None:
            length_cm = float(item["length_cm"])
            if not (MIN_SHRIMP_LENGTH_CM <= length_cm <= MAX_SHRIMP_LENGTH_CM):
                continue
            lengths.append(length_cm)
        item["shrimp_id"] = len(updated) + 1
        updated.append(item)

    avg_length = round(sum(lengths) / len(lengths), 2) if lengths else None
    return updated, avg_length


def render_annotated_image(
    real_img_path: Path,
    output_path: Path,
    measurements: list,
) -> bool:
    """
    Draw YOLOv8 bounding boxes and orientation-aware length measurements onto
    the real image, matching scripts/shrimp_count_length_db_store.py.
    """
    img = cv2.imread(str(real_img_path))
    if img is None:
        return False

    for m in measurements:
        bbox = m.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        length_cm = float(m.get("length_cm", 0.0))

        # Draw bounding box (green, thickness 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw length label above bounding box
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), img))


def find_matching_real_image(storage: ImageStorage, ai_image_name: str) -> Optional[RealImage]:
    """Find the corresponding real image for an AI-processed image."""
    base_name = ai_image_name.replace("count_length_", "")

    for img_type in [ImageType.TEST, ImageType.TRAIN, ImageType.VALIDATION]:
        real_img = storage.session.query(RealImage).filter_by(
            image_name=base_name,
            image_type=img_type,
        ).first()
        if real_img:
            return real_img

    for img_type in [ImageType.TEST, ImageType.TRAIN, ImageType.VALIDATION]:
        real_images = storage.session.query(RealImage).filter_by(image_type=img_type).all()
        ai_base = Path(base_name).stem
        for real_img in real_images:
            real_base = Path(real_img.image_name).stem
            if real_base == ai_base or real_base in ai_base or ai_base in real_base:
                return real_img

    return None


def store_real_folder(folder: Path, image_type: ImageType, label: str):
    """Store all valid (non-LFS-pointer) images from a folder into real_images."""
    if not folder.exists():
        print(f"Directory not found: {folder}")
        return

    image_files = sorted(
        list(folder.glob("*.jpg"))
        + list(folder.glob("*.jpeg"))
        + list(folder.glob("*.png"))
    )

    print("=" * 60)
    print(f"Storing {label} Images to Database ({image_type.value})")
    print("=" * 60)
    print(f"Found {len(image_files)} files\n")

    stored = 0
    skipped_lfs = 0
    errors = 0

    with ImageStorage() as storage:
        for img_path in tqdm(image_files, desc=f"Storing {label}"):
            if is_git_lfs_pointer(img_path):
                skipped_lfs += 1
                continue
            try:
                storage.store_real_image(
                    image_path=str(img_path),
                    image_type=image_type,
                )
                stored += 1
            except Exception as e:
                print(f"\n  Error storing {img_path.name}: {e}")
                errors += 1

    print(f"\n{label} complete: stored/updated={stored}, skipped_lfs={skipped_lfs}, errors={errors}")


def store_count_length_images(force_rerender: bool = False):
    """Regenerate (if LFS pointers or force_rerender) and store all test real + AI bounding-box images."""
    print("Creating database tables if needed...")
    create_tables()

    detections = load_detection_results()
    print(f"Loaded {len(detections)} detection records from {DETECTION_RESULTS_JSON.name}")

    COUNT_LENGTH_DIR.mkdir(parents=True, exist_ok=True)

    # Collect test images to process
    test_images = sorted(
        list(TEST_DIR.glob("*.jpg"))
        + list(TEST_DIR.glob("*.jpeg"))
        + list(TEST_DIR.glob("*.png"))
    )

    if not test_images:
        print(f"No test images found in {TEST_DIR}")
        return

    print("=" * 60)
    print("Storing Real Test Images & AI Bounding-Box Images to Database")
    print("=" * 60)
    print(f"Found {len(test_images)} test images to process\n")

    # Deterministic RNG for consistent pond/cycle/device metadata
    rng = random.Random(42)

    stored_real = 0
    stored_ai = 0
    regenerated = 0
    error_count = 0

    with ImageStorage() as storage:
        for real_path in tqdm(test_images, desc="Processing Real + AI Images"):
            try:
                if is_git_lfs_pointer(real_path):
                    continue

                img_name = real_path.name
                det = detections.get(img_name)
                ai_path = COUNT_LENGTH_DIR / img_name

                measurements, avg_length = (
                    recalc_detection_metrics(det) if det is not None else (None, None)
                )

                # Regenerate AI bounding-box image if missing, LFS pointer, or forced recalculation
                if (force_rerender or not ai_path.exists() or is_git_lfs_pointer(ai_path)) and measurements is not None:
                    ok = render_annotated_image(real_path, ai_path, measurements)
                    if ok:
                        regenerated += 1
                    else:
                        print(f"\n  Failed to render annotated image for {img_name}")
                        error_count += 1
                        continue

                if not ai_path.exists() or is_git_lfs_pointer(ai_path):
                    print(f"\n  Skipping {img_name}: no valid AI annotated image available")
                    error_count += 1
                    continue

                # 1. Store/reuse real test image
                real_img = storage.session.query(RealImage).filter_by(
                    image_name=img_name,
                    image_type=ImageType.TEST,
                ).first()
                if real_img is None:
                    real_img = storage.store_real_image(
                        image_path=str(real_path),
                        image_type=ImageType.TEST,
                        image_name=img_name,
                    )
                stored_real += 1

                # 2. Prepare detection metadata (use filtered measurement count for consistency)
                shrimp_count = len(measurements) if measurements is not None else (det.get("shrimp_count") if det else None)
                if avg_length is not None and avg_length <= 0:
                    avg_length = None
                avg_weight = (
                    storage.calculate_shrimp_weight(avg_length)
                    if avg_length is not None
                    else None
                )

                pond_id = f"pond{rng.randint(1, 4)}"
                cycle_no = f"cycle{rng.randint(1, 4)}"
                device_id = f"D{rng.randint(1, 3)}"
                water_depth_m = rng.choice([5.0, 6.0, 7.0, 10.0])

                # 3. Store AI bounding-box image linked to real_img
                storage.store_ai_image(
                    image_path=str(ai_path),
                    image_type=ImageType.COUNT_LENGTH,
                    real_image_id=real_img.id,
                    shrimp_count=shrimp_count,
                    avg_shrimp_length_cm=avg_length,
                    avg_shrimp_weight_g=avg_weight,
                    measurements=measurements,
                    pond_id=pond_id,
                    cycle_no=cycle_no,
                    device_id=device_id,
                    water_depth_m=water_depth_m,
                    model_used="best.pt",
                    confidence_threshold=0.25,
                    image_name=f"count_length_{img_name}",
                )
                stored_ai += 1

            except Exception as e:
                print(f"\n  Error storing {real_path.name}: {e}")
                error_count += 1

        print("\n" + "=" * 60)
        print("STORAGE SUMMARY")
        print("=" * 60)
        print(f"Annotated AI images regenerated from detection_results.json: {regenerated}")
        print(f"Real test images stored/updated: {stored_real}")
        print(f"AI bounding-box images stored/updated: {stored_ai}")
        print(f"Errors: {error_count}")

        stats = storage.get_image_statistics()
        print("\nDatabase Statistics:")
        print(f"  Total Real Images: {stats['total_real_images']} {stats['real_images_by_type']}")
        print(f"  Total AI Images: {stats['total_ai_images']} {stats['ai_images_by_type']}")
        print("=" * 60)


def store_train_images():
    """Store all training images from data/train/images/."""
    store_real_folder(TRAIN_DIR, ImageType.TRAIN, "Training")


def store_validation_images():
    """Store all validation images from data/valid/images/."""
    store_real_folder(VALID_DIR, ImageType.VALIDATION, "Validation")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Store shrimp detection images to database")
    parser.add_argument("--train", action="store_true", help="Store training images")
    parser.add_argument("--valid", action="store_true", help="Store validation images")
    parser.add_argument("--count-length", action="store_true", help="Store test real + AI count_length images")
    parser.add_argument("--recalc-ai", action="store_true", help="Re-render AI bounding boxes with orientation-aware lengths and update DB")
    parser.add_argument("--all", action="store_true", help="Store all images (test + AI + train + valid)")

    args = parser.parse_args()

    create_tables()

    if args.all or (not args.train and not args.valid and not args.count_length and not args.recalc_ai):
        store_count_length_images(force_rerender=args.recalc_ai)
        print()
        store_train_images()
        print()
        store_validation_images()
    else:
        if args.count_length or args.recalc_ai:
            store_count_length_images(force_rerender=args.recalc_ai)
        if args.train:
            store_train_images()
        if args.valid:
            store_validation_images()


if __name__ == "__main__":
    main()
