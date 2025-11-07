import json
import argparse
import sys
import os
from datetime import datetime
import math

# Add the directory containing the trading_database module to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_database import TradingDatabase

class TradeExporter:
    def __init__(self, db_path="trading_data.db"):
        self.db = TradingDatabase(db_path)
        
    def export_trades_to_json(self, chunk_size=50, output_prefix="trades_export", include_all_data=True):
        """
        Export ALL trades to multiple JSON files, chunked by specified size
        
        Args:
            chunk_size: Number of trades per file
            output_prefix: Prefix for output JSON files
            include_all_data: Whether to include all raw trade data
        """
        try:
            print(f"🔄 Fetching ALL trades from database...")
            
            # Get ALL historical trades
            trades_df = self.db.get_historical_trades(days=36500)  # Large number to get all trades
            
            if trades_df.empty:
                print("❌ No trades found in the database")
                return False
            
            print(f"📊 Found {len(trades_df)} total trades")
            
            # Convert to list of dictionaries
            all_trades_data = []
            
            for idx, trade in trades_df.iterrows():
                trade_dict = trade.to_dict()
                
                # Parse JSON fields if they exist
                if include_all_data:
                    self._parse_json_fields(trade_dict)
                
                all_trades_data.append(trade_dict)
            
            # Calculate number of chunks needed
            total_trades = len(all_trades_data)
            num_chunks = math.ceil(total_trades / chunk_size)
            
            print(f"📦 Splitting {total_trades} trades into {num_chunks} files of {chunk_size} trades each")
            
            successful_exports = 0
            
            for chunk_num in range(num_chunks):
                start_idx = chunk_num * chunk_size
                end_idx = min((chunk_num + 1) * chunk_size, total_trades)
                
                chunk_trades = all_trades_data[start_idx:end_idx]
                
                # Create export structure for this chunk
                export_data = {
                    "export_info": {
                        "export_timestamp": datetime.now().isoformat(),
                        "total_trades_in_chunk": len(chunk_trades),
                        "chunk_number": chunk_num + 1,
                        "total_chunks": num_chunks,
                        "chunk_start_index": start_idx,
                        "chunk_end_index": end_idx - 1,
                        "database_source": self.db.db_path,
                        "export_version": "1.1"
                    },
                    "trades": chunk_trades,
                    "summary": self._generate_summary(chunk_trades)
                }
                
                # Create output filename with chunk number
                output_file = f"{output_prefix}_part{chunk_num + 1:03d}_of_{num_chunks:03d}.json"
                
                # Write to JSON file
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
                
                print(f"✅ Exported chunk {chunk_num + 1}/{num_chunks} to {output_file} ({len(chunk_trades)} trades)")
                successful_exports += 1
            
            # Create a master index file
            self._create_master_index(output_prefix, num_chunks, total_trades, chunk_size)
            
            print(f"\n🎉 Successfully exported {successful_exports} files with {total_trades} total trades")
            print(f"📁 Files prefix: {output_prefix}_part*_of_*.json")
            print(f"📋 Master index: {output_prefix}_MASTER_INDEX.json")
            
            return True
            
        except Exception as e:
            print(f"❌ Error exporting trades: {e}")
            return False
    
    def _create_master_index(self, output_prefix, num_chunks, total_trades, chunk_size):
        """Create a master index file with information about all chunks"""
        master_data = {
            "master_export_info": {
                "export_timestamp": datetime.now().isoformat(),
                "total_trades_exported": total_trades,
                "total_chunks": num_chunks,
                "chunk_size": chunk_size,
                "output_prefix": output_prefix,
                "file_pattern": f"{output_prefix}_part*_of_*.json"
            },
            "chunks": [
                {
                    "chunk_number": i + 1,
                    "filename": f"{output_prefix}_part{i + 1:03d}_of_{num_chunks:03d}.json",
                    "expected_trades": chunk_size if i < num_chunks - 1 else total_trades % chunk_size or chunk_size
                }
                for i in range(num_chunks)
            ]
        }
        
        master_filename = f"{output_prefix}_MASTER_INDEX.json"
        with open(master_filename, 'w', encoding='utf-8') as f:
            json.dump(master_data, f, indent=2, ensure_ascii=False)
        
        return master_filename
    
    def _parse_json_fields(self, trade_dict):
        """Parse JSON string fields into Python objects"""
        json_fields = ['ml_prediction_details', 'technical_indicators_json', 'flags']
        
        for field in json_fields:
            if field in trade_dict and trade_dict[field] and isinstance(trade_dict[field], str):
                try:
                    trade_dict[field] = json.loads(trade_dict[field])
                except json.JSONDecodeError:
                    # Keep as string if not valid JSON
                    pass
    
    def _generate_summary(self, trades_data):
        """Generate summary statistics for the exported trades"""
        if not trades_data:
            return {}
        
        closed_trades = [t for t in trades_data if t.get('exit_price') is not None and not pd.isna(t.get('exit_price'))]
        open_trades = [t for t in trades_data if t.get('exit_price') is None or pd.isna(t.get('exit_price'))]
        
        winning_trades = [t for t in closed_trades if t.get('pnl_percent', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl_percent', 0) < 0]
        
        symbols = list(set(t['symbol'] for t in trades_data if t.get('symbol')))
        
        return {
            "total_trades": len(trades_data),
            "closed_trades": len(closed_trades),
            "open_trades": len(open_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0,
            "symbols_traded": symbols,
            "date_range": {
                "oldest": min(t.get('timestamp') for t in trades_data if t.get('timestamp')),
                "newest": max(t.get('timestamp') for t in trades_data if t.get('timestamp'))
            } if trades_data else {}
        }
    
    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def close(self):
        """Close database connection"""
        if self.db:
            self.db.close()

def main():
    parser = argparse.ArgumentParser(description='Export ALL trading data to multiple JSON files')
    parser.add_argument('--db-path', default='trading_data.db', help='Path to the SQLite database file')
    parser.add_argument('--chunk-size', type=int, default=50, help='Number of trades per JSON file')
    parser.add_argument('--output-prefix', default='trades_export', help='Prefix for output JSON files')
    parser.add_argument('--minimal', action='store_true', help='Export minimal data only (exclude raw JSON fields)')
    
    args = parser.parse_args()
    
    exporter = TradeExporter(args.db_path)
    
    try:
        success = exporter.export_trades_to_json(
            chunk_size=args.chunk_size,
            output_prefix=args.output_prefix,
            include_all_data=not args.minimal
        )
        
        if success:
            print(f"\n🎉 Export completed successfully!")
            print(f"📁 Files: {args.output_prefix}_part*_of_*.json")
        else:
            print("\n💥 Export failed!")
            sys.exit(1)
            
    finally:
        exporter.close()

# Import pandas for the summary generation (moved here to avoid dependency if not needed in main)
import pandas as pd

if __name__ == "__main__":
    main()