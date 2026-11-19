import sys
import os
from pathlib import Path

# Add database directory to path
sys.path.insert(0, str(Path(__file__).parent / "database"))

from models import RealImage, AIImage, ImageType, get_engine
from sqlalchemy.orm import sessionmaker

def get_image_counts():
    """Get total counts of images in database"""
    try:
        engine = get_engine()
        Session = sessionmaker(bind=engine)
        session = Session()
        
        real_count = session.query(RealImage).count()
        ai_count = session.query(AIImage).count()
        
        print(f"📊 Database Image Statistics:")
        print(f"============================")
        print(f"Real images: {real_count}")
        print(f"AI images: {ai_count}")
        print(f"Total images: {real_count + ai_count}")
        print(f"============================")
        
        # Get breakdown by type
        print(f"\n📋 Breakdown by Image Type:")
        print(f"Real Images by Type:")
        for img_type in ImageType:
            count = session.query(RealImage).filter_by(image_type=img_type).count()
            if count > 0:
                print(f"  {img_type.value}: {count}")
        
        print(f"\nAI Images by Type:")
        for img_type in ImageType:
            count = session.query(AIImage).filter_by(image_type=img_type).count()
            if count > 0:
                print(f"  {img_type.value}: {count}")
        
        session.close()
        return real_count, ai_count
        
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return 0, 0

if __name__ == "__main__":
    get_image_counts()