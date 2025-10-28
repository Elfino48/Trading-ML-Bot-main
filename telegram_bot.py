import requests
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
import json

class TelegramBot:
    def __init__(self, bot_token: str, channel_id: str, allowed_user_ids: List[int] = None):
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.allowed_user_ids = allowed_user_ids or []
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.last_update_id = 0
        self.command_handlers = {}
        self.running = False
        self.trading_bot = None
        self._register_commands()
    
    def set_trading_bot(self, trading_bot):
        self.trading_bot = trading_bot
    
    def _register_commands(self):
        self.command_handlers = {
            '/start': self._handle_start,
            '/status': self._handle_status,
            '/portfolio': self._handle_portfolio,
            '/performance': self._handle_performance,
            '/aggressiveness': self._handle_aggressiveness,
            '/pause': self._handle_pause,
            '/resume': self._handle_resume,
            '/stop': self._handle_stop,
            '/trades': self._handle_trades,
            '/symbols': self._handle_symbols,
            '/risk': self._handle_risk,
            '/help': self._handle_help,
            '/errors': self._handle_errors,
            '/emergency': self._handle_emergency,
            '/metrics': self._handle_metrics,
            '/reset_errors': self._handle_reset_errors,
            '/database': self._handle_database
        }
    
    def send_message(self, chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending Telegram message: {e}")
            return False
    
    def send_channel_message(self, text: str, parse_mode: str = "HTML") -> bool:
        return self.send_message(self.channel_id, text, parse_mode)
    
    def get_updates(self):
        try:
            url = f"{self.base_url}/getUpdates"
            params = {
                'offset': self.last_update_id + 1,
                'timeout': 30
            }
            response = requests.get(url, params=params, timeout=35)
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    return data.get('result', [])
            return []
        except Exception as e:
            print(f"Error getting Telegram updates: {e}")
            return []
    
    def process_updates(self, updates):
        for update in updates:
            self.last_update_id = update['update_id']
            if 'message' in update:
                message = update['message']
                chat_id = str(message['chat']['id'])
                user_id = message['from']['id']
                if user_id not in self.allowed_user_ids:
                    self.send_message(chat_id, "❌ Unauthorized. You are not allowed to control this bot.")
                    continue
                if 'text' in message:
                    text = message['text']
                    self.handle_command(chat_id, user_id, text)
    
    def handle_command(self, chat_id: str, user_id: int, text: str):
        command_parts = text.split()
        if not command_parts:
            return
        command = command_parts[0].lower()
        args = command_parts[1:]
        if command in self.command_handlers:
            self.command_handlers[command](chat_id, args)
        else:
            self.send_message(chat_id, f"❌ Unknown command: {command}\nType /help for available commands.")
    
    def _handle_start(self, chat_id: str, args: List[str]):
        message = """
🤖 <b>Trading Bot Control Panel</b>

<b>Available commands:</b>
/status - Bot status and current cycle
/portfolio - Portfolio value and P&L
/performance - Trading performance metrics
/aggressiveness [level] - Change trading aggressiveness
/pause - Pause trading temporarily
/resume - Resume trading
/stop - Stop the bot completely
/trades - Recent trade history
/symbols - List of trading symbols
/risk - Current risk metrics

<b>New Enhanced Commands:</b>
/errors - Show current error status
/emergency - Emergency protocols status
/metrics - Advanced performance metrics
/reset_errors - Reset error handler
/database - Database statistics

/help - Show this help message

<b>Aggressiveness Levels:</b>
• conservative - Safe, fewer trades
• moderate - Balanced (recommended) 
• aggressive - Higher frequency
• high - Maximum risk
        """
        self.send_message(chat_id, message.strip())
    
    def _handle_status(self, chat_id: str, args: List[str]):
        if not self.trading_bot:
            self.send_message(chat_id, "❌ Bot not connected")
            return
        try:
            portfolio_value = self.trading_bot.get_portfolio_value()
            risk_summary = self.trading_bot.risk_manager.get_risk_summary()
            cycle_count = getattr(self.trading_bot, 'cycle_count', 0)
            error_summary = self.trading_bot.get_error_summary()
            emergency_status = self.trading_bot.get_emergency_status()
            message = f"""
📊 <b>BOT STATUS</b>

🔄 <b>Cycle:</b> #{cycle_count}
💰 <b>Portfolio:</b> ${portfolio_value:,.2f}
📈 <b>Daily P&L:</b> {risk_summary.get('daily_pnl_percent', 0):+.2f}%
🎯 <b>Aggressiveness:</b> {self.trading_bot.aggressiveness.upper()}
📈 <b>Trades Today:</b> {risk_summary.get('trades_today', 0)}

<b>System Health:</b>
🛡️ <b>Error Status:</b> {error_summary.get('health_status', 'UNKNOWN')}
🚨 <b>Emergency Mode:</b> {'ACTIVE' if emergency_status.get('emergency_mode') else 'Inactive'}
📊 <b>Total Errors:</b> {error_summary.get('total_errors', 0)}

🛡️ <b>Max Daily Loss:</b> {risk_summary.get('max_daily_loss', 0)}%

⏰ <i>Last update: {datetime.now().strftime('%H:%M:%S')}</i>
            """
            self.send_message(chat_id, message.strip())
        except Exception as e:
            self.send_message(chat_id, f"❌ Error getting status: {e}")
    
    def _handle_errors(self, chat_id: str, args: List[str]):
        if not self.trading_bot:
            self.send_message(chat_id, "❌ Bot not connected")
            return
        try:
            error_summary = self.trading_bot.get_error_summary()
            message = f"""
🛡️ <b>ERROR STATUS</b>

<b>Health Status:</b> {error_summary.get('health_status', 'UNKNOWN')}
<b>Total Errors:</b> {error_summary.get('total_errors', 0)}
<b>API Errors:</b> {error_summary.get('api_errors', 0)}
<b>Trading Errors:</b> {error_summary.get('trading_errors', 0)}
<b>ML Errors:</b> {error_summary.get('ml_errors', 0)}
<b>Circuit Breaker:</b> {'ACTIVE' if error_summary.get('circuit_breaker_active') else 'Inactive'}

<b>Recent Errors:</b>
"""
            recent_errors = error_summary.get('recent_errors', [])
            for error in recent_errors[-5:]:
                error_time = error['timestamp'].strftime('%H:%M') if hasattr(error['timestamp'], 'strftime') else 'Unknown'
                message += f"• {error_time} - {error['type']}: {error['error'][:50]}...\n"
            if not recent_errors:
                message += "• No recent errors ✅\n"
            message += f"\n⏰ <i>{datetime.now().strftime('%H:%M:%S')}</i>"
            self.send_message(chat_id, message.strip())
        except Exception as e:
            self.send_message(chat_id, f"❌ Error getting error status: {e}")
    
    def _handle_emergency(self, chat_id: str, args: List[str]):
        if not self.trading_bot:
            self.send_message(chat_id, "❌ Bot not connected")
            return
        try:
            emergency_status = self.trading_bot.get_emergency_status()
            message = f"""
🆘 <b>EMERGENCY PROTOCOLS</b>

<b>Emergency Mode:</b> {'🚨 ACTIVE' if emergency_status.get('emergency_mode') else '✅ Inactive'}
"""
            if emergency_status.get('emergency_mode'):
                message += f"<b>Start Time:</b> {emergency_status.get('emergency_start_time').strftime('%H:%M:%S')}\n"
                message += f"<b>Triggered Protocols:</b> {len(emergency_status.get('triggered_protocols', []))}\n"
            message += f"""
<b>Current Metrics:</b>
• Consecutive Losses: {emergency_status.get('current_metrics', {}).get('consecutive_losses', 0)}
• Daily High Balance: ${emergency_status.get('current_metrics', {}).get('daily_high_balance', 0):,.2f}

<b>Thresholds:</b>
• Max Drawdown: {emergency_status.get('thresholds', {}).get('max_drawdown', 0)*100:.1f}%
• Max Daily Loss: {emergency_status.get('thresholds', {}).get('max_daily_loss', 0)*100:.1f}%
• Max Consecutive Losses: {emergency_status.get('thresholds', {}).get('max_consecutive_losses', 0)}

"""
            if args and args[0] == 'reset':
                self.trading_bot.reset_emergency_mode("Telegram command")
                message += "\n✅ Emergency mode reset\n"
            message += f"⏰ <i>{datetime.now().strftime('%H:%M:%S')}</i>"
            self.send_message(chat_id, message.strip())
        except Exception as e:
            self.send_message(chat_id, f"❌ Error getting emergency status: {e}")
    
    def _handle_metrics(self, chat_id: str, args: List[str]):
        if not self.trading_bot:
            self.send_message(chat_id, "❌ Bot not connected")
            return
        try:
            stats = self.trading_bot.database.get_trading_statistics(days=7)
            message = """
📊 <b>ADVANCED METRICS</b>

<b>7-Day Performance:</b>
"""
            if stats:
                message += f"• Total Trades: {stats.get('total_trades', 0)}\n"
                message += f"• Win Rate: {stats.get('win_rate', 0):.1f}%\n"
                message += f"• Avg PnL: {stats.get('avg_pnl', 0):.2f}%\n"
                message += f"• Total PnL: {stats.get('total_pnl', 0):.2f}%\n"
                if 'best_trade' in stats:
                    message += f"• Best Trade: {stats['best_trade'].get('pnl_percent', 0):.2f}% ({stats['best_trade'].get('symbol', 'N/A')})\n"
                if 'worst_trade' in stats:
                    message += f"• Worst Trade: {stats['worst_trade'].get('pnl_percent', 0):.2f}% ({stats['worst_trade'].get('symbol', 'N/A')})\n"
                message += "\n<b>Top Performing Symbols:</b>\n"
                symbol_perf = stats.get('symbol_performance', [])
                for symbol in symbol_perf[:3]:
                    message += f"• {symbol.get('symbol')}: {symbol.get('avg_pnl', 0):.2f}% ({symbol.get('trade_count', 0)} trades)\n"
            else:
                message += "• No trading data available\n"
            db_stats = self.trading_bot.database.get_trading_statistics(days=30)
            if db_stats and 'total_trades' in db_stats:
                message += f"\n<b>30-Day Total Trades:</b> {db_stats['total_trades']}"
            message += f"\n⏰ <i>{datetime.now().strftime('%H:%M:%S')}</i>"
            self.send_message(chat_id, message.strip())
        except Exception as e:
            self.send_message(chat_id, f"❌ Error getting metrics: {e}")
    
    def _handle_reset_errors(self, chat_id: str, args: List[str]):
        if not self.trading_bot:
            self.send_message(chat_id, "❌ Bot not connected")
            return
        try:
            self.trading_bot.reset_error_handler()
            self.send_message(chat_id, "✅ Error handler circuit breaker reset")
        except Exception as e:
            self.send_message(chat_id, f"❌ Error resetting error handler: {e}")
    
    def _handle_database(self, chat_id: str, args: List[str]):
        if not self.trading_bot:
            self.send_message(chat_id, "❌ Bot not connected")
            return
        try:
            trades_7d = self.trading_bot.database.get_historical_trades(days=7)
            trades_30d = self.trading_bot.database.get_historical_trades(days=30)
            performance_history = self.trading_bot.database.get_performance_history(days=7)
            message = f"""
💾 <b>DATABASE STATISTICS</b>

<b>Trade Records:</b>
• Last 7 days: {len(trades_7d)} trades
• Last 30 days: {len(trades_30d)} trades

<b>Performance Records:</b>
• Last 7 days: {len(performance_history)} records

<b>System Events:</b>
• Recent events logged and monitored
• Error tracking active
• Performance metrics stored

<b>Database Health:</b> ✅ Active
"""
            if len(trades_7d) > 0:
                win_rate = (len(trades_7d[trades_7d['pnl_percent'] > 0]) / len(trades_7d)) * 100
                message += f"• 7-Day Win Rate: {win_rate:.1f}%\n"
            message += f"\n⏰ <i>{datetime.now().strftime('%H:%M:%S')}</i>"
            self.send_message(chat_id, message.strip())
        except Exception as e:
            self.send_message(chat_id, f"❌ Error getting database stats: {e}")
    
    def _handle_portfolio(self, chat_id: str, args: List[str]):
        if not self.trading_bot:
            self.send_message(chat_id, "❌ Bot not connected")
            return
        try:
            portfolio_value = self.trading_bot.get_portfolio_value()
            risk_summary = self.trading_bot.risk_manager.get_risk_summary()
            daily_pnl = risk_summary.get('daily_pnl_percent', 0)
            daily_start = risk_summary.get('daily_start_balance', portfolio_value)
            message = f"""
💰 <b>PORTFOLIO</b>

<b>Current Value:</b> ${portfolio_value:,.2f}
<b>Daily Start:</b> ${daily_start:,.2f}
<b>Today's P&L:</b> ${portfolio_value - daily_start:+.2f}
<b>Today's P&L %:</b> {daily_pnl:+.2f}%

<b>Risk Summary:</b>
• Max Position: ${risk_summary.get('max_position_size', 0)}
• Max Daily Loss: {risk_summary.get('max_daily_loss', 0)}%
• Current VaR: ${risk_summary.get('current_var', 0):.2f}

⏰ <i>{datetime.now().strftime('%H:%M:%S')}</i>
            """
            self.send_message(chat_id, message.strip())
        except Exception as e:
            self.send_message(chat_id, f"❌ Error getting portfolio: {e}")
    
    def _handle_performance(self, chat_id: str, args: List[str]):
        if not self.trading_bot:
            self.send_message(chat_id, "❌ Bot not connected")
            return
        try:
            performance = self.trading_bot.get_performance_summary()
            message = f"""
📊 <b>PERFORMANCE SUMMARY</b>

<b>Total Trades:</b> {performance.get('total_trades', 0)}
<b>Win Rate:</b> {performance.get('win_rate', 0):.1f}%
<b>Average Confidence:</b> {performance.get('avg_confidence', 0):.1f}%
<b>Average Risk/Reward:</b> {performance.get('avg_risk_reward', 0):.1f}
<b>Aggressiveness:</b> {performance.get('aggressiveness', 'N/A').upper()}
"""
            try:
                recent_trades = self.trading_bot.execution_engine.get_trade_history(limit=5)
                if recent_trades:
                    message += f"\n<b>Last {len(recent_trades)} Trades:</b>\n"
                    for trade in recent_trades[-5:]:
                        success_emoji = "✅" if trade.get('success', False) else "❌"
                        action_emoji = "🟢" if trade.get('side') == 'Buy' else "🔴"
                        message += f"{success_emoji}{action_emoji} {trade.get('symbol', 'N/A')} - ${trade.get('size_usdt', 0):.2f}\n"
            except:
                pass
            if 'recent_stats' in performance:
                stats = performance['recent_stats']
                if stats:
                    message += f"\n<b>7-Day Performance:</b>\n"
                    message += f"• Trades: {stats.get('total_trades', 0)}\n"
                    message += f"• Win Rate: {stats.get('win_rate', 0):.1f}%\n"
                    message += f"• Total PnL: {stats.get('total_pnl', 0):.2f}%\n"
                    symbol_perf = stats.get('symbol_performance', [])
                    if symbol_perf:
                        best_symbol = symbol_perf[0]
                        message += f"• Best Symbol: {best_symbol.get('symbol')} ({best_symbol.get('avg_pnl', 0):.2f}%)\n"
            message += f"\n⏰ <i>{datetime.now().strftime('%H:%M:%S')}</i>"
            self.send_message(chat_id, message.strip())
        except Exception as e:
            self.send_message(chat_id, f"❌ Error getting performance: {e}")
    
    def _handle_aggressiveness(self, chat_id: str, args: List[str]):
        if not self.trading_bot:
            self.send_message(chat_id, "❌ Bot not connected")
            return
        if not args:
            self.send_message(chat_id, "Usage: /aggressiveness [conservative|moderate|aggressive|high]")
            return
        new_level = args[0].lower()
        if self.trading_bot.change_aggressiveness(new_level):
            self.send_message(chat_id, f"✅ Aggressiveness changed to {new_level.upper()}")
        else:
            self.send_message(chat_id, "❌ Invalid aggressiveness level")
    
    def _handle_trades(self, chat_id: str, args: List[str]):
        if not self.trading_bot:
            self.send_message(chat_id, "❌ Bot not connected")
            return
        try:
            trades = self.trading_bot.execution_engine.get_trade_history(limit=10)
            message = "📋 <b>RECENT TRADES</b>\n\n"
            for trade in trades:
                emoji = "🟢" if trade.get('success', False) else "🔴"
                message += f"{emoji} {trade['symbol']} {trade['side']} - ${trade.get('size_usdt', 0):.2f}\n"
            self.send_message(chat_id, message)
        except Exception as e:
            self.send_message(chat_id, f"❌ Error getting trades: {e}")
    
    def _handle_pause(self, chat_id: str, args: List[str]):
        if not self.trading_bot:
            self.send_message(chat_id, "❌ Bot not connected")
            return
        try:
            if hasattr(self.trading_bot, 'paused'):
                self.trading_bot.paused = True
                self.send_message(chat_id, "⏸️ Trading paused. Bot will continue monitoring but won't execute trades.")
            else:
                self.send_message(chat_id, "❌ Pause functionality not implemented in current version")
        except Exception as e:
            self.send_message(chat_id, f"❌ Error pausing bot: {e}")
    
    def _handle_resume(self, chat_id: str, args: List[str]):
        if not self.trading_bot:
            self.send_message(chat_id, "❌ Bot not connected")
            return
        try:
            if hasattr(self.trading_bot, 'paused'):
                self.trading_bot.paused = False
                self.send_message(chat_id, "▶️ Trading resumed. Bot will execute trades in next cycle.")
            else:
                self.send_message(chat_id, "❌ Resume functionality not implemented in current version")
        except Exception as e:
            self.send_message(chat_id, f"❌ Error resuming bot: {e}")
    
    def _handle_stop(self, chat_id: str, args: List[str]):
        if not self.trading_bot:
            self.send_message(chat_id, "❌ Bot not connected")
            return
        try:
            if args and args[0] == 'confirm':
                emergency_result = self.trading_bot.execution_engine.emergency_stop_with_verification()
                message = "🛑 <b>BOT STOPPED</b>\n\n"
                if emergency_result.get('success'):
                    message += "✅ All positions closed successfully\n"
                else:
                    message += "⚠️ Some positions may still be open\n"
                message += "🤖 Trading bot has been stopped.\n"
                message += "⏰ Use the start script to restart the bot."
                self.send_message(chat_id, message)
                if hasattr(self.trading_bot, 'running'):
                    self.trading_bot.running = False
            else:
                self.send_message(
                    chat_id,
                    "⚠️ <b>CONFIRM BOT STOP</b>\n\n"
                    "This will:\n"
                    "• Close all open positions\n" 
                    "• Cancel all pending orders\n"
                    "• Stop the trading bot completely\n\n"
                    "Type <code>/stop confirm</code> to proceed."
                )
        except Exception as e:
            self.send_message(chat_id, f"❌ Error stopping bot: {e}")
    
    def _handle_symbols(self, chat_id: str, args: List[str]):
        try:
            from config import SYMBOLS
            message = "📊 <b>TRADING SYMBOLS</b>\n\n"
            message += f"<b>Total Symbols:</b> {len(SYMBOLS)}\n"
            message += f"<b>Timeframe:</b> {getattr(self.trading_bot, 'timeframe', '15')}min\n\n"
            symbols_per_line = 4
            for i in range(0, len(SYMBOLS), symbols_per_line):
                line_symbols = SYMBOLS[i:i + symbols_per_line]
                message += " • " + "  ".join(line_symbols) + "\n"
            message += f"\n🔄 <i>Monitoring all symbols in real-time</i>"
            self.send_message(chat_id, message)
        except Exception as e:
            self.send_message(chat_id, f"❌ Error getting symbols: {e}")
    
    def _handle_risk(self, chat_id: str, args: List[str]):
        if not self.trading_bot:
            self.send_message(chat_id, "❌ Bot not connected")
            return
        try:
            risk_summary = self.trading_bot.risk_manager.get_risk_summary()
            risk_assessment = self.trading_bot.risk_manager.get_risk_assessment()
            message = "🛡️ <b>RISK MANAGEMENT</b>\n\n"
            message += f"<b>Daily P&L:</b> {risk_summary.get('daily_pnl_percent', 0):+.2f}%\n"
            message += f"<b>Max Daily Loss:</b> {risk_summary.get('max_daily_loss', 0)}%\n"
            message += f"<b>Current Exposure:</b> ${risk_summary.get('current_exposure', 0):.2f}\n"
            message += f"<b>Exposure %:</b> {risk_summary.get('exposure_percent', 0):.1f}%\n"
            message += f"<b>Consecutive Losses:</b> {risk_summary.get('consecutive_losses', 0)}\n"
            message += f"<b>Circuit Breaker:</b> {'🔴 ACTIVE' if risk_summary.get('circuit_breaker') else '🟢 Inactive'}\n\n"
            if risk_assessment and 'overall_risk_level' in risk_assessment:
                risk_level = risk_assessment['overall_risk_level']
                risk_emoji = "🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"
                message += f"<b>Overall Risk Level:</b> {risk_emoji} {risk_level}\n"
            concentration = risk_assessment.get('concentration', {})
            if concentration:
                message += f"<b>Diversification Score:</b> {concentration.get('diversification_score', 0):.1f}/100\n"
                message += f"<b>Positions:</b> {concentration.get('position_count', 0)}\n\n"
            health_checks = risk_assessment.get('health_checks', {})
            if health_checks:
                message += "<b>Health Checks:</b>\n"
                message += f"• Daily P&L: {'✅' if health_checks.get('daily_pnl_ok') else '❌'}\n"
                message += f"• Circuit Breaker: {'✅' if health_checks.get('circuit_breaker_ok') else '❌'}\n"
                message += f"• Consecutive Losses: {'✅' if health_checks.get('consecutive_losses_ok') else '❌'}\n"
                message += f"• Exposure: {'✅' if health_checks.get('exposure_ok') else '❌'}\n"
                message += f"• Concentration: {'✅' if health_checks.get('concentration_ok') else '❌'}\n\n"
            if risk_assessment.get('overall_risk_level') == "HIGH":
                suggestions = self.trading_bot.risk_manager.get_risk_reduction_suggestions()
                if suggestions:
                    message += "<b>Risk Reduction Suggestions:</b>\n"
                    for suggestion in suggestions[:3]:
                        message += f"• {suggestion}\n"
            message += f"\n⏰ <i>{datetime.now().strftime('%H:%M:%S')}</i>"
            self.send_message(chat_id, message)
        except Exception as e:
            self.send_message(chat_id, f"❌ Error getting risk metrics: {e}")
    
    def _handle_help(self, chat_id: str, args: List[str]):
        message = """
🤖 <b>Advanced Trading Bot - Command Help</b>

<b>Basic Commands:</b>
/status - Bot status and current cycle
/portfolio - Portfolio value and P&L  
/performance - Trading performance metrics
/aggressiveness [level] - Change trading aggressiveness
/trades - Recent trade history
/symbols - List of trading symbols
/risk - Current risk metrics

<b>Control Commands:</b>
/pause - Pause trading temporarily
/resume - Resume trading
/stop - Stop the bot completely (with confirmation)

<b>Enhanced Monitoring:</b>
/errors - Show current error status
/emergency - Emergency protocols status  
/metrics - Advanced performance metrics
/reset_errors - Reset error handler
/database - Database statistics

<b>Aggressiveness Levels:</b>
• conservative - Safe, fewer trades (min confidence: 35%)
• moderate - Balanced (min confidence: 25%) 
• aggressive - Higher frequency (min confidence: 20%)
• high - Maximum risk (min confidence: 15%)

<b>Risk Management:</b>
• Circuit breaker activates after 3 consecutive losses
• Maximum daily loss: 5-15% (based on aggressiveness)
• Position sizing uses Kelly Criterion
• Portfolio correlation limits enforced

💡 <i>All commands require authorization. Use /start for quick overview.</i>
        """
        self.send_message(chat_id, message.strip())
    
    def start_polling(self):
        self.running = True
        print("🤖 Telegram command bot started...")
        while self.running:
            try:
                updates = self.get_updates()
                if updates:
                    self.process_updates(updates)
                time.sleep(1)
            except Exception as e:
                print(f"Error in Telegram polling: {e}")
                time.sleep(5)
    
    def stop_polling(self):
        self.running = False
    
    def log_bot_start(self, portfolio_value: float, symbols: List[str], aggressiveness: str):
        message = f"""
🤖 <b>TRADING BOT STARTED</b>

💰 <b>Portfolio:</b> ${portfolio_value:,.2f}
📊 <b>Symbols:</b> {', '.join(symbols)}
🎯 <b>Aggressiveness:</b> {aggressiveness.upper()}
🛡️ <b>Error Handler:</b> Active
💾 <b>Database:</b> Active
🆘 <b>Emergency Protocols:</b> Active
🕒 <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<i>Bot is now running and monitoring markets...</i>
        """
        self.send_channel_message(message.strip())
    
    def log_bot_stop(self, portfolio_value: float, performance: Dict = None):
        message = f"""
🛑 <b>TRADING BOT STOPPED</b>

💰 <b>Final Portfolio:</b> ${portfolio_value:,.2f}
        """
        if performance:
            message += f"""
📈 <b>Performance Summary:</b>
   • Total Trades: {performance.get('total_trades', 0)}
   • Win Rate: {performance.get('win_rate', 0):.1f}%
   • Avg Confidence: {performance.get('avg_confidence', 0):.1f}%
   • Aggressiveness: {performance.get('aggressiveness', 'N/A').upper()}
            """
        message += f"\n🕒 <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.send_channel_message(message.strip())
    
    def log_trade_execution(self, trade_data: Dict):
        symbol = trade_data['symbol']
        action = trade_data['action']
        quantity = trade_data['quantity']
        price = trade_data['current_price']
        size_usdt = trade_data['position_size']
        confidence = trade_data['confidence']
        stop_loss = trade_data['stop_loss']
        take_profit = trade_data['take_profit']
        risk_reward = trade_data['risk_reward_ratio']
        aggressiveness = trade_data.get('aggressiveness', 'N/A')
        emoji = "🟢" if action == 'BUY' else "🔴"
        message = f"""
{emoji} <b>TRADE EXECUTED</b>

<b>Symbol:</b> {symbol}
<b>Action:</b> {action}
<b>Quantity:</b> {quantity:.4f}
<b>Price:</b> ${price:,.2f}
<b>Size:</b> ${size_usdt:,.2f}
<b>Confidence:</b> {confidence:.1f}%
<b>Aggressiveness:</b> {aggressiveness.upper()}

<b>Risk Management:</b>
   • Stop Loss: ${stop_loss:,.2f}
   • Take Profit: ${take_profit:,.2f}
   • Risk/Reward: {risk_reward:.1f}:1

🕒 <i>{datetime.now().strftime('%H:%M:%S')}</i>
        """
        self.send_channel_message(message.strip())
    
    def log_trade_error(self, symbol: str, action: str, error: str):
        message = f"""
❌ <b>TRADE ERROR</b>

<b>Symbol:</b> {symbol}
<b>Action:</b> {action}
<b>Error:</b> <code>{error}</code>

🕒 <i>{datetime.now().strftime('%H:%M:%S')}</i>
        """
        self.send_channel_message(message.strip())
    
    def log_cycle_summary(self, summary: Dict):
        trades_executed = summary['trades_executed']
        strong_signals = summary['strong_signals']
        moderate_signals = summary.get('moderate_signals', 0)
        portfolio_value = summary['portfolio_value']
        pnl_percent = summary['pnl_percent']
        aggressiveness = summary.get('aggressiveness', 'N/A')
        message = f"""
📊 <b>CYCLE SUMMARY</b>

<b>Trades Executed:</b> {trades_executed}
<b>Strong Signals:</b> {strong_signals}
<b>Moderate Signals:</b> {moderate_signals}
<b>Portfolio Value:</b> ${portfolio_value:,.2f}
<b>P&L:</b> {pnl_percent:+.2f}%
<b>Aggressiveness:</b> {aggressiveness.upper()}

🕒 <i>{datetime.now().strftime('%H:%M:%S')}</i>
        """
        self.send_channel_message(message.strip())
    
    def log_ml_training(self, symbol: str, accuracy: float, message_detail: Optional[str] = None):
        message = f"""
🧠 <b>ML MODEL TRAINED</b>

<b>Symbol:</b> {symbol}
<b>Accuracy:</b> {accuracy:.1f}%
"""
        if message_detail:
                    message += f"<b>Note:</b> {message_detail}\n"
        message += f"""
🕒 <i>{datetime.now().strftime('%H:%M:%S')}</i>
        """
        self.send_channel_message(message.strip())
    
    def log_error(self, error: str, context: str = ""):
        message = f"""
🚨 <b>ERROR</b>

<b>Context:</b> {context}
<b>Error:</b> <code>{error}</code>

🕒 <i>{datetime.now().strftime('%H:%M:%S')}</i>
        """
        self.send_channel_message(message.strip())
    
    def log_important_event(self, title: str, message: str):
        formatted_message = f"""
📢 <b>{title}</b>

{message}

🕒 <i>{datetime.now().strftime('%H:%M:%S')}</i>
        """
        self.send_channel_message(formatted_message.strip())