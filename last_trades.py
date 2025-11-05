import json
import argparse
import sys
import os
from datetime import datetime

# Add the directory containing the trading_database module to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_database import TradingDatabase

class TradeExporter:
    def __init__(self, db_path="trading_data.db"):
        self.db = TradingDatabase(db_path)
        
    def export_trades_to_json(self, n_trades=100, output_file="trades_export.json", include_all_data=True):
        """
        Export the last N trades to a JSON file
        
        Args:
            n_trades: Number of recent trades to export
            output_file: Output JSON file path
            include_all_data: Whether to include all raw trade data
        """
        try:
            print(f"🔄 Fetching last {n_trades} trades from database...")
            
            # Get historical trades (this method returns the most recent first)
            trades_df = self.db.get_historical_trades(days=365)  # Get up to 1 year of trades
            
            if trades_df.empty:
                print("❌ No trades found in the database")
                return False
            
            # Take the last N trades (already sorted by timestamp DESC)
            recent_trades = trades_df.head(n_trades)
            
            print(f"📊 Found {len(recent_trades)} trades")
            
            # Convert to list of dictionaries
            trades_data = []
            
            for idx, trade in recent_trades.iterrows():
                trade_dict = trade.to_dict()
                
                # Parse JSON fields if they exist
                if include_all_data:
                    self._parse_json_fields(trade_dict)
                
                trades_data.append(trade_dict)
            
            # Create export structure
            export_data = {
                "export_info": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_trades_exported": len(trades_data),
                    "database_source": self.db.db_path,
                    "export_version": "1.0"
                },
                "trades": trades_data,
                "summary": self._generate_summary(trades_data)
            }
            
            # Write to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
            
            print(f"✅ Successfully exported {len(trades_data)} trades to {output_file}")
            print(f"📈 Summary: {export_data['summary']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error exporting trades: {e}")
            return False
    
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
    parser = argparse.ArgumentParser(description='Export trading data to JSON')
    parser.add_argument('--db-path', default='trading_data.db', help='Path to the SQLite database file')
    parser.add_argument('--n-trades', type=int, default=20, help='Number of recent trades to export')
    parser.add_argument('--output', default='trades_export.json', help='Output JSON file path')
    parser.add_argument('--minimal', action='store_true', help='Export minimal data only (exclude raw JSON fields)')
    
    args = parser.parse_args()
    
    exporter = TradeExporter(args.db_path)
    
    try:
        success = exporter.export_trades_to_json(
            n_trades=args.n_trades,
            output_file=args.output,
            include_all_data=not args.minimal
        )
        
        if success:
            print(f"\n🎉 Export completed successfully!")
            print(f"📁 File: {args.output}")
        else:
            print("\n💥 Export failed!")
            sys.exit(1)
            
    finally:
        exporter.close()

# Import pandas for the summary generation (moved here to avoid dependency if not needed in main)
import pandas as pd

if __name__ == "__main__":
    main()