from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def detect_shrimp(model_path, source, save_dir='../runs/detect/predict'):
    """
    Detect shrimp in images/videos - Shows ONLY bounding boxes
    
    Args:
        model_path: Path to trained model weights
        source: Path to image, video, or directory
        save_dir: Directory to save results
    """
    # Load trained model
    model = YOLO(model_path)
    
    # Run detection
    results = model.predict(
        source=source,
        conf=0.25,              # confidence threshold
        iou=0.45,               # NMS IOU threshold
        save=True,              # save results
        save_txt=True,          # save labels
        save_conf=True,         # save confidence scores
        project=save_dir,       # save directory
        name='exp',             # experiment name
        exist_ok=True,          # overwrite existing
        show_labels=False,      # ❌ HIDE class labels (shrimp name)
        show_conf=False,        # ❌ HIDE confidence scores
        line_width=2,           # bounding box thickness
    )
    
    print(f"\n✅ Detection completed!")
    print(f"Results saved to: {save_dir}/exp")
    
    return results

def detect_and_display(model_path, image_path):
    """
    Detect and display results immediately - Shows ONLY bounding boxes
    """
    model = YOLO(model_path)
    results = model(image_path)
    
    # Display results
    for result in results:
        # Get annotated image (with custom settings)
        annotated_img = result.plot(
            labels=False,       # ❌ Don't show class labels
            conf=False,         # ❌ Don't show confidence scores
            line_width=2,       # Bounding box thickness
        )
        
        # Display
        cv2.imshow('Shrimp Detection', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Print detections (console only, not on image)
        print(f"\nDetections in {image_path}:")
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            print(f"  - Class: {result.names[cls]}, Confidence: {conf:.2f}, Box: {coords}")

if __name__ == '__main__':
    # Example usage
    MODEL_PATH = 'runs/runs/detect/train/weights/best.pt'
    
    # Detect on single image
    # detect_shrimp(MODEL_PATH, '../test_images/shrimp1.jpg')
    
    # Detect on directory of images
    detect_shrimp(MODEL_PATH, '../test_images')
    
    # Detect on video
    # detect_shrimp(MODEL_PATH, '../test_video.mp4')
    
    # Detect and display immediately
    # detect_and_display(MODEL_PATH, '../test_images/shrimp1.jpg')
    
    # print("Uncomment the desired detection mode above!")