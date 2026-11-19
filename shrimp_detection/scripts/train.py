"""
YOLOv8 Shrimp Detection - GPU-Optimized Training Script (FIXED PATHS)
Dataset: 247 train + 27 valid images
Hardware: Windows with GPU, 12GB RAM
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path

def check_gpu():
    """Check if GPU is available and print details"""
    print("\n" + "="*50)
    print("🔍 GPU CHECK")
    print("="*50)
    
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"✅ Number of GPUs: {torch.cuda.device_count()}")
        device = 'cuda'
    else:
        print("❌ No GPU detected! Training will use CPU (much slower)")
        print("⚠️  Make sure you have:")
        print("   1. NVIDIA GPU installed")
        print("   2. CUDA drivers installed")
        print("   3. PyTorch with CUDA support installed")
        device = 'cpu'
    
    print("="*50 + "\n")
    return device

def find_data_yaml():
    """Find the correct path to data.yaml"""
    # Get current directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Possible paths to check
    possible_paths = [
        Path('../data/data.yaml'),           # From scripts/ folder
        Path('data/data.yaml'),              # From root folder
        Path('../data.yaml'),                # One level up
        Path('../../data/data.yaml'),        # Two levels up
    ]
    
    # Check each path
    for path in possible_paths:
        abs_path = path.resolve()
        if abs_path.exists():
            print(f"✅ Found data.yaml at: {abs_path}")
            return str(abs_path)
    
    # If not found, show error with helpful info
    print("\n" + "="*50)
    print("❌ ERROR: data.yaml NOT FOUND!")
    print("="*50)
    print("\nLooked in these locations:")
    for path in possible_paths:
        print(f"  ❌ {path.resolve()}")
    
    print("\n📋 Please make sure:")
    print("  1. Your Roboflow dataset is copied to: ../data/")
    print("  2. The file 'data.yaml' exists in: ../data/data.yaml")
    print("  3. You're running this script from: shrimp_detection/scripts/")
    print("\n💡 Current folder structure should be:")
    print("  shrimp_detection/")
    print("  ├── data/")
    print("  │   ├── train/")
    print("  │   ├── valid/")
    print("  │   └── data.yaml  ← This file must exist!")
    print("  └── scripts/")
    print("      └── train.py  ← You are here")
    print("="*50 + "\n")
    
    raise FileNotFoundError("data.yaml not found. Please copy your dataset first!")

def main():
    # Check GPU availability
    device = check_gpu()
    
    # Find data.yaml
    data_yaml_path = find_data_yaml()
    
    print("="*50)
    print("🦐 YOLOV8 SHRIMP DETECTION - GPU TRAINING")
    print("="*50)
    print(f"📊 Dataset: 247 train + 27 valid images")
    print(f"🖥️  Device: {device.upper()}")
    print(f"📦 Model: YOLOv8n (Nano - Fast & Accurate)")
    print(f"📄 Config: {data_yaml_path}")
    print("="*50 + "\n")
    
    # Load pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    print("🚀 Starting training...\n")
    
    # Train the model with GPU-optimized settings
    results = model.train(
        # Dataset - USE ABSOLUTE PATH
        data=data_yaml_path,                # FIXED: Use absolute path
        
        # Training duration
        epochs=150,
        patience=30,
        
        # Image settings
        imgsz=640,
        
        # Batch size - ADJUSTED FOR GPU
        batch=16,                           # Reduce to 8 if OOM error
        
        # Hardware
        device=device,
        workers=8,
        
        # Optimization
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        cos_lr=True,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        
        # Saving
        save=True,
        save_period=10,
        project='../runs/detect',
        name='train',
        exist_ok=True,
        
        # Validation
        val=True,
        
        # Logging
        verbose=True,
        seed=42,
        deterministic=True,
        
        # Advanced
        pretrained=True,
        amp=True,
        fraction=1.0,
    )
    
    print("\n" + "="*50)
    print("✅ TRAINING COMPLETED!")
    print("="*50)
    print(f"📁 Results saved to: runs/detect/train/")
    print(f"🏆 Best weights: runs/detect/train/weights/best.pt")
    print(f"📊 Last weights: runs/detect/train/weights/last.pt")
    print(f"📈 Metrics plot: runs/detect/train/results.png")
    print("="*50 + "\n")

if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Fix: Copy your Roboflow dataset to the 'data' folder!")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)