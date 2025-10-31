import sqlite3
import os
import sys
from datetime import datetime
import time

# --- CONFIGURATION ---
DB_FILE = "trading_data.db"
# --- END CONFIGURATION ---

def get_db_creation_time_iso(db_path: str) -> str:
    """
    Gets the database file's creation time and returns it as an ISO string.
    """
    try:
        if not os.path.exists(db_path):
            print(f"Error: Database file not found at {db_path}", file=sys.stderr)
            return None
            
        ctime_float = os.path.getctime(db_path)
        creation_datetime = datetime.fromtimestamp(ctime_float)
        creation_iso_str = creation_datetime.isoformat()
        
        print(f"Database file creation time: {creation_iso_str}")
        return creation_iso_str
        
    except Exception as e:
        print(f"Error getting database creation time: {e}", file=sys.stderr)
        return None

def main():
    print(f"--- Database Cleanup Tool ---")
    print(f"WARNING: This script will permanently modify the database '{DB_FILE}'")
    print("... Starting cleanup in 3 seconds. Press Ctrl+C to cancel. ...")
    time.sleep(3) # Give the user a moment to cancel

    conn = None
    try:
        # --- 1. Get DB Creation Time ---
        creation_time_iso = get_db_creation_time_iso(DB_FILE)
        if creation_time_iso is None:
            return

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # --- 2. Delete All Records Older Than DB Creation Time ---
        print(f"\n[Step 1/3] Deleting all records created before {creation_time_iso}...")
        
        # Map tables to their correct timestamp column
        tables_to_clean = {
            'trades': 'timestamp', 
            'closure_events': 'timestamp', 
            'performance_metrics': 'timestamp', 
            'system_events': 'timestamp',
            'ml_model_performance': 'timestamp', 
            'market_data': 'timestamp', 
            'prediction_quality': 'timestamp',
            'feature_importance': 'timestamp', 
            'model_training_history': 'training_date', # This one is different
            'model_drift_detection': 'timestamp'
        }
        
        total_old_rows_deleted = 0
        for table, time_column in tables_to_clean.items():
            try:
                cursor.execute(f"DELETE FROM {table} WHERE {time_column} < ?", (creation_time_iso,))
                rows_deleted = cursor.rowcount
                print(f"  - Deleted {rows_deleted} old records from '{table}'")
                total_old_rows_deleted += rows_deleted
            except sqlite3.Error as e:
                print(f"  - Error cleaning table '{table}': {e} (continuing...)")

        print(f"Total old records deleted: {total_old_rows_deleted}")

        # --- 3. Remove All Records from Bybit Closure Table ---
        print("\n[Step 2/3] Deleting ALL records from 'closure_events'...")
        cursor.execute("DELETE FROM closure_events")
        print(f"  - Deleted {cursor.rowcount} records from 'closure_events'")

        # --- 4. Reset All Opened Records ---
        print("\n[Step 3/3] Resetting all successful 'trades' records to 'open' state...")
        cursor.execute("""
            UPDATE trades
            SET 
                exit_price = NULL,
                pnl_usdt = NULL,
                pnl_percent = NULL,
                exit_reason = NULL,
                outcome_updated = 0,
                closure_event_id = NULL
            WHERE 
                success = 'True'
        """)
        print(f"  - Reset {cursor.rowcount} trade records.")

        # --- 5. Commit Changes ---
        conn.commit()
        print("\nAll operations complete. Database cleanup and reset finished.")
        print("You can now restart the bot to begin reconciliation.")

    except sqlite3.Error as e:
        print(f"\nAn SQL error occurred: {e}", file=sys.stderr)
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()