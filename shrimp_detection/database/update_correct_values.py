"""
Update Database with Correct Shrimp Count and Length Values
This script runs detection on images and updates the database with correct values
"""

import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import db_config
from models import AIImage, ImageType, RealImage


def update_ai_image_values(image_id, correct_count, correct_length, session):
    """Update AI image with correct shrimp count and average length"""
    ai_img = session.query(AIImage).filter_by(id=image_id).first()
    
    if ai_img:
        old_count = ai_img.shrimp_count
        old_length = ai_img.avg_shrimp_length_cm
        
        ai_img.shrimp_count = correct_count
        ai_img.avg_shrimp_length_cm = correct_length
        
        session.commit()
        
        print(f"Updated AI Image ID {image_id}:")
        print(f"  Count: {old_count} -> {correct_count}")
        print(f"  Avg Length: {old_length} -> {correct_length}")
        return True
    
    return False


def show_current_values():
    """Display current values in database"""
    engine = create_engine(db_config.connection_string)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        print("="*60)
        print("Current AI Images in Database")
        print("="*60)
        
        ai_images = session.query(AIImage).filter_by(image_type=ImageType.COUNT_LENGTH).all()
        
        for img in ai_images:
            print(f"\nID: {img.id}")
            print(f"  Name: {img.image_name}")
            print(f"  Shrimp Count: {img.shrimp_count}")
            print(f"  Avg Length: {img.avg_shrimp_length_cm} cm")
            if img.measurements_json:
                print(f"  Measurements: {img.measurements_json[:100]}...")
        
        print(f"\nTotal AI Images: {len(ai_images)}")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        session.close()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update shrimp detection values in database")
    parser.add_argument("--show", action="store_true", help="Show current values")
    parser.add_argument("--update", type=int, help="Update specific image ID")
    parser.add_argument("--count", type=int, help="New shrimp count")
    parser.add_argument("--length", type=float, help="New average length")
    
    args = parser.parse_args()
    
    if args.show:
        show_current_values()
    elif args.update and args.count is not None and args.length is not None:
        engine = create_engine(db_config.connection_string)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            success = update_ai_image_values(args.update, args.count, args.length, session)
            if success:
                print("Update successful!")
            else:
                print(f"Image ID {args.update} not found")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            session.close()
    else:
        print("Usage:")
        print("  python update_correct_values.py --show")
        print("  python update_correct_values.py --update <id> --count <count> --length <length>")


if __name__ == "__main__":
    main()
