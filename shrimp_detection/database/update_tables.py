"""
Update Database Tables - Add new columns for weight and metadata
"""

from sqlalchemy import create_engine, text
from config import db_config


def update_tables():
    """Add new columns to ai_images table"""
    engine = create_engine(db_config.connection_string)
    
    with engine.connect() as conn:
        # Add new columns
        columns_to_add = [
            ("avg_shrimp_weight_g", "FLOAT"),
            ("pond_id", "VARCHAR(50)"),
            ("cycle_no", "VARCHAR(50)"),
            ("device_id", "VARCHAR(50)"),
            ("water_depth_m", "FLOAT"),
        ]
        
        for col_name, col_type in columns_to_add:
            try:
                conn.execute(text(f"ALTER TABLE ai_images ADD COLUMN IF NOT EXISTS {col_name} {col_type}"))
                print(f"✅ Added column: {col_name}")
            except Exception as e:
                print(f"⚠️ Column {col_name} may already exist: {e}")
        
        conn.commit()
        print("\n✅ Database tables updated successfully!")


if __name__ == "__main__":
    update_tables()
