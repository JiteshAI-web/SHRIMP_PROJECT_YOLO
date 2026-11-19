"""
Create and verify PostgreSQL indexes for Shrimp Detection System.
Dramatically improves query speed for dashboard image loading, charts, and stats.
"""

import sys
from pathlib import Path
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import get_engine


INDEX_STATEMENTS = [
    # Real images lookups by image_type & chronological sorting
    (
        "ix_real_images_image_type",
        "CREATE INDEX IF NOT EXISTS ix_real_images_image_type ON real_images (image_type)",
    ),
    (
        "ix_real_images_created_at",
        "CREATE INDEX IF NOT EXISTS ix_real_images_created_at ON real_images (created_at)",
    ),
    # Foreign key join from ai_images -> real_images
    (
        "ix_ai_images_real_image_id",
        "CREATE INDEX IF NOT EXISTS ix_ai_images_real_image_id ON ai_images (real_image_id)",
    ),
    # Composite index for filtering COUNT_LENGTH and sorting by created_at
    (
        "ix_ai_images_type_created_at",
        "CREATE INDEX IF NOT EXISTS ix_ai_images_type_created_at ON ai_images (image_type, created_at DESC)",
    ),
    # Partial index for chart_data & latest detection queries (shrimp_count IS NOT NULL)
    (
        "ix_ai_images_count_created_at",
        "CREATE INDEX IF NOT EXISTS ix_ai_images_count_created_at ON ai_images (created_at ASC) WHERE shrimp_count IS NOT NULL",
    ),
]


def create_indexes():
    engine = get_engine()
    with engine.begin() as conn:
        for name, stmt in INDEX_STATEMENTS:
            conn.execute(text(stmt))
            print(f"OK: ensured index {name}")

        # Refresh PostgreSQL query planner statistics
        conn.execute(text("ANALYZE real_images"))
        conn.execute(text("ANALYZE ai_images"))
        print("OK: ANALYZE completed on real_images and ai_images")

        # List all active indexes
        rows = conn.execute(
            text(
                """
                SELECT tablename, indexname, indexdef
                FROM pg_indexes
                WHERE tablename IN ('real_images', 'ai_images')
                ORDER BY tablename, indexname
                """
            )
        ).fetchall()

        print("\nActive PostgreSQL Indexes:")
        for tbl, idx, idxdef in rows:
            print(f"  [{tbl}] {idx}")
            print(f"      {idxdef}")


if __name__ == "__main__":
    create_indexes()
