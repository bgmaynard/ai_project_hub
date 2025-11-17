"""
Database Migration: Add ui_type column to user_layouts table
This migration adds UI type isolation support to the layouts table
"""

import sqlite3
import os
from pathlib import Path

DB_PATH = Path(__file__).parent / "warrior_trading.db"


def migrate():
    """Add ui_type column to user_layouts table"""
    print(f"Migrating database: {DB_PATH}")

    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        print("Schema will be created with ui_type column on first run")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if ui_type column already exists
        cursor.execute("PRAGMA table_info(user_layouts)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'ui_type' in columns:
            print("[OK] ui_type column already exists - no migration needed")
            return

        print("Adding ui_type column to user_layouts table...")

        # Add the ui_type column with default value 'monitor'
        cursor.execute("""
            ALTER TABLE user_layouts
            ADD COLUMN ui_type TEXT DEFAULT 'monitor'
        """)

        # Update all existing layouts to have ui_type = 'monitor'
        cursor.execute("""
            UPDATE user_layouts
            SET ui_type = 'monitor'
            WHERE ui_type IS NULL
        """)

        conn.commit()
        print("[OK] Migration completed successfully!")
        print("  - Added ui_type column")
        print("  - Set existing layouts to 'monitor' type")

    except Exception as e:
        print(f"[FAIL] Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Database Migration: UI Type Isolation")
    print("=" * 60)
    migrate()
    print("=" * 60)
