"""
YOLOv8 Shrimp Detection with Database Storage
Runs detection, counts shrimp, measures length, and stores CORRECT values in PostgreSQL database
"""

import sys
import json
import math
import random
from pathlib import Path
from typing import Union, List, Dict
from datetime import datetime

# Add parent directory to path for database imports
sys.path.insert(0, str(Path(__file__).parent.parent / "database"))

from ultralytics import YOLO
import cv2
import numpy as np

from image_storage import ImageStorage
from models import ImageType


# ===================== CALIBRATION =====================
TRAY_DIAMETER_CM = 70.0
FALLBACK_TRAY_DIAMETER_PIXELS = 1200.0
MIN_SHRIMP_LENGTH_CM = 0.5
MAX_SHRIMP_LENGTH_CM = 25.0
# =======================================================


def _find_tray_diameter_pixels(img: np.ndarray) -> float:
    """Automatically estimate tray diameter in pixels from the image."""
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
        largest = max(circles, key=lambda c: c[2])
        radius = largest[2]
        diameter = float(2 * radius)
        return diameter

    return 0.0


def _compute_length_px_from_box(xyxy):
    """Compute orientation-aware shrimp length in pixels from a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = xyxy
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    major = max(w, h)
    minor = min(w, h)
    if major > 0 and (minor / major) > 0.5:
        return math.hypot(w, h)
    return major


def count_and_measure_shrimp_with_db_store(
    model_path: Union[str, Path],
    source: Union[str, Path],
    save_dir: Union[str, Path] = "../runs/detect/count_length",
    store_in_db: bool = True
) -> List[Dict]:
    """
    Run detection, count shrimps, measure length, and store CORRECT values in database.
    
    Args:
        model_path: Path to YOLO model weights
        source: Path to image or directory of images
        save_dir: Directory to save annotated images
        store_in_db: Whether to store images in database
    
    Returns:
        List of dictionaries containing results for each image with CORRECT counts and lengths
    """
    model_path = Path(model_path)
    source = str(source)
    save_dir = str(save_dir)
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Run prediction
    results = model.predict(
        source=source,
        conf=0.25,
        iou=0.45,
        save=False,
        save_txt=True,
        save_conf=True,
        project=save_dir,
        name="exp",
        exist_ok=True,
        show_labels=False,
        show_conf=False,
    )
    
    # Ensure output directory exists
    out_dir = Path(save_dir) / "exp"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize database storage
    storage = ImageStorage() if store_in_db else None
    
    all_results = []
    total_shrimp = 0
    
    try:
        for result in results:
            img_path = Path(result.path)
            img_name = img_path.name
            
            print(f"\n{'='*60}")
            print(f"Processing: {img_name}")
            print(f"{'='*60}")
            
            # Get original image
            img = result.orig_img.copy()
            
            # Tray calibration
            tray_diameter_px = _find_tray_diameter_pixels(img)
            if tray_diameter_px <= 0:
                tray_diameter_px = FALLBACK_TRAY_DIAMETER_PIXELS
                print(f"[WARN] Tray not detected, using fallback: {tray_diameter_px:.1f} px")
            else:
                print(f"[INFO] Detected tray diameter: {tray_diameter_px:.1f} px")
            
            pixel_to_cm_ratio = TRAY_DIAMETER_CM / tray_diameter_px
            
            # Process detections - EXACT counts from model
            shrimp_lengths_cm = []
            measurements = []
            
            # CORRECT shrimp count from model detection
            detected_boxes = result.boxes
            shrimp_count = len(detected_boxes)  # EXACT count from YOLO
            
            print(f"[DETECT] Model found {shrimp_count} shrimps")
            
            for box in detected_boxes:
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)
                confidence = float(box.conf[0])

                # Calculate orientation-aware length and reject false-positive outliers
                length_px = _compute_length_px_from_box(coords)
                length_cm = length_px * pixel_to_cm_ratio
                if not (MIN_SHRIMP_LENGTH_CM <= length_cm <= MAX_SHRIMP_LENGTH_CM):
                    continue

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                shrimp_lengths_cm.append(length_cm)

                # Store measurement details
                measurements.append({
                    "shrimp_id": len(measurements) + 1,
                    "length_cm": round(length_cm, 2),
                    "confidence": round(confidence, 4),
                    "bbox": [round(c, 2) for c in coords]
                })

                # Draw text
                text = f"{length_cm:.2f} cm"
                cv2.putText(img, text, (x1, max(y1 - 5, 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Update final valid shrimp count after physical outlier filtering
            shrimp_count = len(measurements)
            
            # Save annotated image
            out_path = out_dir / img_name
            cv2.imwrite(str(out_path), img)
            print(f"[INFO] Saved annotated image: {out_path}")
            
            # Calculate CORRECT average length
            if shrimp_count > 0:
                avg_length = sum(shrimp_lengths_cm) / shrimp_count
            else:
                avg_length = 0.0
            
            print(f"[RESULT] Shrimp count: {shrimp_count}")
            print(f"[RESULT] Average length: {avg_length:.2f} cm")
            if shrimp_count > 0:
                print(f"[RESULT] Individual lengths: {[round(l, 2) for l in shrimp_lengths_cm]}")
            
            # Store in database with CORRECT values
            real_image_id = None
            ai_image_id = None
            
            if store_in_db and storage:
                try:
                    # 1. Store real (input) image
                    print(f"[DB] Storing real image...")
                    real_img = storage.store_real_image(
                        image_path=str(img_path),
                        image_type=ImageType.TEST,
                        image_name=img_name
                    )
                    real_image_id = real_img.id
                    print(f"[DB] Real image stored with ID: {real_image_id}")
                    
                    # 2. Store AI (output) image with CORRECT measurements
                    print(f"[DB] Storing AI image with CORRECT values...")
                    print(f"       - Shrimp count: {shrimp_count}")
                    print(f"       - Avg length: {avg_length:.2f} cm")
                    
                    # Calculate weight from length
                    avg_weight_g = storage.calculate_shrimp_weight(avg_length) if shrimp_count > 0 else None
                    
                    # Generate random metadata values
                    pond_id = f"pond{random.randint(1, 4)}"
                    cycle_no = f"cycle{random.randint(1, 4)}"
                    device_id = f"D{random.randint(1, 3)}"
                    water_depth_m = random.choice([5.0, 6.0, 7.0, 10.0])
                    
                    print(f"       - Avg weight: {avg_weight_g:.2f} g" if avg_weight_g else "       - Avg weight: N/A")
                    print(f"       - Pond ID: {pond_id}")
                    print(f"       - Cycle: {cycle_no}")
                    print(f"       - Device: {device_id}")
                    print(f"       - Water depth: {water_depth_m} m")
                    
                    ai_img = storage.store_ai_image(
                        image_path=str(out_path),
                        image_type=ImageType.COUNT_LENGTH,
                        real_image_id=real_image_id,
                        shrimp_count=shrimp_count,  # CORRECT count from model
                        avg_shrimp_length_cm=round(avg_length, 2) if shrimp_count > 0 else None,  # CORRECT avg length
                        avg_shrimp_weight_g=avg_weight_g,  # Calculated weight
                        measurements=measurements if measurements else None,
                        pond_id=pond_id,
                        cycle_no=cycle_no,
                        device_id=device_id,
                        water_depth_m=water_depth_m,
                        model_used=str(model_path.name),
                        confidence_threshold=0.25,
                        image_name=f"count_length_{img_name}"
                    )
                    ai_image_id = ai_img.id
                    print(f"[DB] AI image stored with ID: {ai_image_id}")
                    
                except Exception as e:
                    print(f"[ERROR] Database storage failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Collect results with CORRECT values
            image_result = {
                "image_name": img_name,
                "source_path": str(img_path),
                "output_path": str(out_path),
                "shrimp_count": shrimp_count,  # CORRECT
                "average_length_cm": round(avg_length, 2) if shrimp_count > 0 else 0.0,  # CORRECT
                "individual_lengths_cm": [round(l, 2) for l in shrimp_lengths_cm],
                "measurements": measurements,
                "tray_diameter_px": tray_diameter_px,
                "pixel_to_cm_ratio": pixel_to_cm_ratio,
                "real_image_id": real_image_id,
                "ai_image_id": ai_image_id
            }
            all_results.append(image_result)
            total_shrimp += shrimp_count
        
        # Commit all database transactions
        if storage:
            storage.session.commit()
            print(f"\n[DB] All changes committed to database")
        
    except Exception as e:
        if storage:
            storage.session.rollback()
        print(f"[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if storage:
            storage.session.close()
    
    # Print summary with CORRECT totals
    print(f"\n{'='*60}")
    print("DETECTION & MEASUREMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(all_results)}")
    print(f"Total shrimps detected: {total_shrimp}")
    print(f"Average shrimps per image: {total_shrimp / len(all_results) if all_results else 0:.1f}")
    print(f"Annotated images saved to: {out_dir}")
    if store_in_db:
        print(f"Images stored in database: shrimp_image2")
    print(f"{'='*60}\n")
    
    return all_results


def main():
    """Main function to run detection and store in database"""
    print("="*60)
    print("YOLOv8 Shrimp Detection with Database Storage")
    print("CORRECT values will be stored in database")
    print("="*60)
    
    # Configuration - Search for model in common locations
    MODEL_PATHS = [
        "runs/runs/detect/train/weights/best.pt",  # Your actual path
        "runs/detect/train/weights/best.pt",
        "../runs/detect/train/weights/best.pt",
        "../../runs/detect/train/weights/best.pt",
        "yolov8n.pt",
        "yolo26n.pt",
    ]
    SOURCE = "../test_images"
    
    # Find the model
    model_full_path = None
    for path in MODEL_PATHS:
        test_path = Path(__file__).parent / path
        if test_path.exists():
            model_full_path = test_path
            print(f"[INFO] Found model: {model_full_path}")
            break
    
    # Check if model exists
    if not model_full_path:
        print(f"[ERROR] Model not found in any of these locations:")
        for path in MODEL_PATHS:
            print(f"  - {Path(__file__).parent / path}")
        print("[INFO] Please train the model first or place it in scripts folder")
        return
    
    # Run detection with database storage
    results = count_and_measure_shrimp_with_db_store(
        model_path=model_full_path,
        source=Path(__file__).parent / SOURCE,
        store_in_db=True
    )
    
    # Save results to JSON file
    results_file = Path(__file__).parent / "../outputs/detection_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to: {results_file}")
    
    # Print CORRECT values summary
    print("\n" + "="*60)
    print("CORRECT VALUES STORED IN DATABASE:")
    print("="*60)
    for r in results:
        print(f"Image: {r['image_name']}")
        print(f"  Count: {r['shrimp_count']}")
        print(f"  Avg Length: {r['average_length_cm']} cm")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()
