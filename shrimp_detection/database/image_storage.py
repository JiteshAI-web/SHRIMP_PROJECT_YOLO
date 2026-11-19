"""
Image Storage Utility for Shrimp Detection System
Handles storing and retrieving images from PostgreSQL database
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, BinaryIO
from datetime import datetime
from PIL import Image
import io

from sqlalchemy.orm import Session
from models import RealImage, AIImage, ImageType, get_session_maker


class ImageStorage:
    """Utility class for storing and retrieving images from database"""
    
    def __init__(self, session: Session = None):
        """
        Initialize ImageStorage
        
        Args:
            session: SQLAlchemy session. If None, creates a new session.
        """
        if session is None:
            Session = get_session_maker()
            self.session = Session()
            self._own_session = True
        else:
            self.session = session
            self._own_session = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._own_session:
            if exc_type is None:
                self.session.commit()
            else:
                self.session.rollback()
            self.session.close()
    
    def _get_image_info(self, image_path: str) -> Tuple[bytes, int, int, int, str]:
        """
        Extract image information from file
        
        Returns:
            Tuple of (binary_data, width, height, file_size, file_format)
        """
        path = Path(image_path)
        
        # Read binary data
        with open(path, 'rb') as f:
            binary_data = f.read()
        
        # Get image dimensions
        with Image.open(path) as img:
            width, height = img.size
            file_format = img.format.lower() if img.format else path.suffix.lower().replace('.', '')
        
        file_size = len(binary_data)
        
        return binary_data, width, height, file_size, file_format
    
    def store_real_image(
        self,
        image_path: str,
        image_type: ImageType,
        image_name: Optional[str] = None
    ) -> RealImage:
        """
        Store a real/original image in the database
        
        Args:
            image_path: Path to the image file
            image_type: Type of image (train, validation, test, predict)
            image_name: Custom name for the image. If None, uses filename.
        
        Returns:
            RealImage object
        """
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Get image info
        binary_data, width, height, file_size, file_format = self._get_image_info(image_path)
        
        # Use provided name or filename
        name = image_name or path.name
        
        # Check if image already exists
        existing = self.session.query(RealImage).filter_by(
            image_name=name,
            image_type=image_type
        ).first()
        
        if existing:
            print(f"⚠️ Image '{name}' of type '{image_type}' already exists. Updating...")
            existing.image_data = binary_data
            existing.file_size_bytes = file_size
            existing.width = width
            existing.height = height
            existing.updated_at = datetime.utcnow()
            self.session.commit()
            return existing
        
        # Create new record
        real_image = RealImage(
            image_name=name,
            image_type=image_type,
            image_data=binary_data,
            file_format=file_format,
            file_size_bytes=file_size,
            width=width,
            height=height,
            source_path=str(path.absolute())
        )
        
        self.session.add(real_image)
        self.session.commit()
        
        print(f"✅ Stored real image: {name} (ID: {real_image.id}, Type: {image_type})")
        return real_image
    
    def calculate_shrimp_weight(self, length_cm: float) -> float:
        """
        Calculate shrimp weight in grams using the formula:
        W = 0.0108 × L^2.6935
        where W is weight in grams and L is length in cm
        """
        if length_cm is None or length_cm <= 0:
            return None
        weight_g = 0.0108 * (length_cm ** 2.6935)
        return round(weight_g, 2)

    def store_ai_image(
        self,
        image_path: str,
        image_type: ImageType,
        real_image_id: int,
        shrimp_count: Optional[int] = None,
        avg_shrimp_length_cm: Optional[float] = None,
        avg_shrimp_weight_g: Optional[float] = None,
        measurements: Optional[List[dict]] = None,
        pond_id: Optional[str] = None,
        cycle_no: Optional[str] = None,
        device_id: Optional[str] = None,
        water_depth_m: Optional[float] = None,
        model_used: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        processing_time_ms: Optional[int] = None,
        image_name: Optional[str] = None
    ) -> AIImage:
        """
        Store an AI-processed image in the database
        
        Args:
            image_path: Path to the AI-processed image file
            image_type: Type of image (count_length, detect, predict)
            real_image_id: ID of the corresponding real image
            shrimp_count: Number of shrimps detected
            avg_shrimp_length_cm: Average shrimp length in cm
            measurements: List of individual measurements as dicts
            model_used: Name of the model used
            confidence_threshold: Confidence threshold used
            processing_time_ms: Processing time in milliseconds
            image_name: Custom name for the image
        
        Returns:
            AIImage object
        """
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Verify real image exists
        real_image = self.session.query(RealImage).filter_by(id=real_image_id).first()
        if not real_image:
            raise ValueError(f"Real image with ID {real_image_id} not found")
        
        # Get image info
        binary_data, width, height, file_size, file_format = self._get_image_info(image_path)
        
        # Use provided name or filename
        name = image_name or path.name
        
        # Check if image already exists
        existing = self.session.query(AIImage).filter_by(
            image_name=name,
            image_type=image_type
        ).first()
        
        measurements_json = json.dumps(measurements) if measurements else None
        
        # Calculate weight if length provided and weight not provided
        if avg_shrimp_weight_g is None and avg_shrimp_length_cm is not None and avg_shrimp_length_cm > 0:
            avg_shrimp_weight_g = self.calculate_shrimp_weight(avg_shrimp_length_cm)
        
        if existing:
            print(f"⚠️ AI image '{name}' of type '{image_type}' already exists. Updating...")
            existing.image_data = binary_data
            existing.file_size_bytes = file_size
            existing.width = width
            existing.height = height
            existing.shrimp_count = shrimp_count
            existing.avg_shrimp_length_cm = avg_shrimp_length_cm
            existing.avg_shrimp_weight_g = avg_shrimp_weight_g
            existing.measurements_json = measurements_json
            existing.pond_id = pond_id
            existing.cycle_no = cycle_no
            existing.device_id = device_id
            existing.water_depth_m = water_depth_m
            existing.model_used = model_used
            existing.confidence_threshold = confidence_threshold
            existing.processing_time_ms = processing_time_ms
            existing.updated_at = datetime.utcnow()
            self.session.commit()
            return existing
        
        # Create new record
        ai_image = AIImage(
            real_image_id=real_image_id,
            image_name=name,
            image_type=image_type,
            image_data=binary_data,
            file_format=file_format,
            file_size_bytes=file_size,
            width=width,
            height=height,
            shrimp_count=shrimp_count,
            avg_shrimp_length_cm=avg_shrimp_length_cm,
            avg_shrimp_weight_g=avg_shrimp_weight_g,
            measurements_json=measurements_json,
            pond_id=pond_id,
            cycle_no=cycle_no,
            device_id=device_id,
            water_depth_m=water_depth_m,
            model_used=model_used,
            confidence_threshold=confidence_threshold,
            processing_time_ms=processing_time_ms,
            source_path=str(path.absolute())
        )
        
        self.session.add(ai_image)
        self.session.commit()
        
        print(f"✅ Stored AI image: {name} (ID: {ai_image.id}, Type: {image_type}, Count: {shrimp_count})")
        return ai_image
    
    def get_real_image(self, image_id: int) -> Optional[RealImage]:
        """Get a real image by ID"""
        return self.session.query(RealImage).filter_by(id=image_id).first()
    
    def get_ai_image(self, image_id: int) -> Optional[AIImage]:
        """Get an AI image by ID"""
        return self.session.query(AIImage).filter_by(id=image_id).first()
    
    def get_real_image_by_name(self, image_name: str, image_type: ImageType) -> Optional[RealImage]:
        """Get a real image by name and type"""
        return self.session.query(RealImage).filter_by(
            image_name=image_name,
            image_type=image_type
        ).first()
    
    def get_ai_images_by_real_image(self, real_image_id: int) -> List[AIImage]:
        """Get all AI images associated with a real image"""
        return self.session.query(AIImage).filter_by(real_image_id=real_image_id).all()
    
    def save_image_to_file(self, image_id: int, output_path: str, is_ai_image: bool = False) -> str:
        """
        Save an image from database to file
        
        Args:
            image_id: ID of the image
            output_path: Path to save the image
            is_ai_image: True if AI image, False if real image
        
        Returns:
            Path to saved file
        """
        if is_ai_image:
            image = self.get_ai_image(image_id)
        else:
            image = self.get_real_image(image_id)
        
        if not image:
            raise ValueError(f"Image with ID {image_id} not found")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(image.image_data)
        
        print(f"✅ Saved image to: {output_path}")
        return str(output_path)
    
    def get_image_statistics(self) -> dict:
        """Get statistics about stored images"""
        real_count = self.session.query(RealImage).count()
        ai_count = self.session.query(AIImage).count()
        
        real_by_type = {}
        for img_type in ImageType:
            count = self.session.query(RealImage).filter_by(image_type=img_type).count()
            if count > 0:
                real_by_type[img_type.value] = count
        
        ai_by_type = {}
        for img_type in ImageType:
            count = self.session.query(AIImage).filter_by(image_type=img_type).count()
            if count > 0:
                ai_by_type[img_type.value] = count
        
        return {
            "total_real_images": real_count,
            "total_ai_images": ai_count,
            "real_images_by_type": real_by_type,
            "ai_images_by_type": ai_by_type
        }
    
    def delete_real_image(self, image_id: int) -> bool:
        """Delete a real image and its associated AI images"""
        image = self.get_real_image(image_id)
        if image:
            self.session.delete(image)
            self.session.commit()
            print(f"✅ Deleted real image ID {image_id} and associated AI images")
            return True
        return False
    
    def delete_ai_image(self, image_id: int) -> bool:
        """Delete an AI image"""
        image = self.get_ai_image(image_id)
        if image:
            self.session.delete(image)
            self.session.commit()
            print(f"✅ Deleted AI image ID {image_id}")
            return True
        return False


# Convenience functions for quick usage
def store_train_image(image_path: str, image_name: Optional[str] = None) -> RealImage:
    """Store a training image"""
    with ImageStorage() as storage:
        return storage.store_real_image(image_path, ImageType.TRAIN, image_name)


def store_validation_image(image_path: str, image_name: Optional[str] = None) -> RealImage:
    """Store a validation image"""
    with ImageStorage() as storage:
        return storage.store_real_image(image_path, ImageType.VALIDATION, image_name)


def store_test_image(image_path: str, image_name: Optional[str] = None) -> RealImage:
    """Store a test image"""
    with ImageStorage() as storage:
        return storage.store_real_image(image_path, ImageType.TEST, image_name)


def store_predict_image(image_path: str, image_name: Optional[str] = None) -> RealImage:
    """Store a prediction input image"""
    with ImageStorage() as storage:
        return storage.store_real_image(image_path, ImageType.PREDICT, image_name)


def store_count_length_image(
    image_path: str,
    real_image_id: int,
    shrimp_count: int,
    avg_length_cm: float,
    **kwargs
) -> AIImage:
    """Store a count & length measurement image"""
    with ImageStorage() as storage:
        return storage.store_ai_image(
            image_path=image_path,
            image_type=ImageType.COUNT_LENGTH,
            real_image_id=real_image_id,
            shrimp_count=shrimp_count,
            avg_shrimp_length_cm=avg_length_cm,
            **kwargs
        )


def store_detect_image(image_path: str, real_image_id: int, **kwargs) -> AIImage:
    """Store a detection result image"""
    with ImageStorage() as storage:
        return storage.store_ai_image(
            image_path=image_path,
            image_type=ImageType.DETECT,
            real_image_id=real_image_id,
            **kwargs
        )


if __name__ == "__main__":
    # Test the image storage
    print("Testing Image Storage...")
    
    with ImageStorage() as storage:
        stats = storage.get_image_statistics()
        print(f"\n📊 Current Statistics:")
        print(f"  Real Images: {stats['total_real_images']}")
        print(f"  AI Images: {stats['total_ai_images']}")
