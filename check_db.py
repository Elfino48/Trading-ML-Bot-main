import sqlite3
import pandas as pd
import os

DB_FILE = "trading_data.db"

def check_database():
    print(f"--- Checking Database '{DB_FILE}' ---")
    
    if not os.path.exists(DB_FILE):
        print(f"❌ ERROR: Database file '{DB_FILE}' does not exist in this directory.")
        print("This is the problem. Your bot is running in a different folder.")
        return

    try:
        con = sqlite3.connect(DB_FILE)
        
        # 1. Check performance_metrics table
        print("\nChecking 'performance_metrics' (for PnL Chart)...")
        try:
            df_perf = pd.read_sql_query("SELECT * FROM performance_metrics ORDER BY date DESC LIMIT 5", con)
            if df_perf.empty:
                print("⚠️  'performance_metrics' table is EMPTY.")
                print("   (This is why the PnL chart is blank)")
            else:
                print(f"✅  Found {len(df_perf)} performance records. (Bot is writing data)")
                print(df_perf)
        except Exception as e:
            print(f"❌ ERROR reading 'performance_metrics': {e}")

        # 2. Check trades table
        print("\nChecking 'trades' (for Trade List)...")
        try:
            df_trades = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 5", con)
            if df_trades.empty:
                print("⚠️  'trades' table is EMPTY.")
                print("   (This is why the trade list is blank)")
            else:
                print(f"✅  Found {len(df_trades)} trade records. (Bot is writing data)")
                print(df_trades[['timestamp', 'symbol', 'action', 'success', 'error_message']].head())
        except Exception as e:
            print(f"❌ ERROR reading 'trades': {e}")
            
        con.close()

    except Exception as e:
        print(f"❌ CRITICAL ERROR connecting to database: {e}")

if __name__ == "__main__":
    check_database()