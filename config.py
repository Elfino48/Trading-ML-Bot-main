import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Bybit Configuration with Environment Variables
BYBIT_CONFIG = {
    "API_KEY": os.getenv("BYBIT_API_KEY", "86ZGMVwycRYeuBYRk6"),
    "API_SECRET": os.getenv("BYBIT_API_SECRET", "5auq3d3Dmfl4VSrTgjCuqKvRdni8NeYywQBJ"),
    "BASE_URL": "https://api.bybit.com",
    "WS_PUBLIC_URL": "wss://stream.bybit.com/v5/public/linear",
    "WS_PRIVATE_URL": "wss://stream.bybit.com/v5/private",
}

# Telegram Configuration
TELEGRAM_CONFIG = {
    "BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "7409712157:AAHi-TnRYbrVeWx81fJbsVKiDxL8Zqcsios"),
    "CHANNEL_ID": os.getenv("TELEGRAM_CHANNEL_ID", "-1002530068871"),
    "ALLOWED_USER_IDS": [int(x) for x in os.getenv("TELEGRAM_ALLOWED_USER_IDS", "397802701").split(",")],
    "ENABLED": os.getenv("TELEGRAM_ENABLED", "True").lower() == "true",
    "LOG_LEVEL": os.getenv("TELEGRAM_LOG_LEVEL", "ALL")
}

DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Trading Parameters
SYMBOLS = os.getenv("TRADING_SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,SOLUSDT,DOGEUSDT").split(",")
TIMEFRAME = os.getenv("TRADING_TIMEFRAME", "15")
LEVERAGE = int(os.getenv("LEVERAGE", "10"))


# Add this near your other bot parameters (like DEBUG_MODE)
EMERGENCY_PROTOCOLS_ENABLED = os.getenv("EMERGENCY_PROTOCOLS_ENABLED", "True").lower() == "true"
# --- NEW: RISK MULTIPLIER ---
# Set this to 1.0 for normal trading.
# Set to 10.0 to use 10x your equity for position sizing and exposure calculations.
# WARNING: This significantly increases risk and leverage.
RISK_MULTIPLIER = float(os.getenv("RISK_MULTIPLIER", "60.0"))

# Risk Configuration
class RiskConfig:
    LEVELS = {
        "conservative": {
            "min_confidence": 30,
            "max_position_size_usdt": 100,
            "max_daily_loss_percent": 5,
            "global_stop_loss_percent": 2,
            "base_size_percent": 0.02,
            "kelly_multiplier": 0.1,
            "sl_atr_multiple": 1.8,
            "tp_atr_multiple": 2.5,
            "max_sl_percent": 2,
            "buy_threshold": 20,
            "sell_threshold": -20,
            "strong_threshold": 40,
            "strategy_weights": {
                'trend_following': 0.40,
                'mean_reversion': 0.30,
                'breakout': 0.20,
                'ml_prediction': 0.10
            },
            "risk_multiplier": RISK_MULTIPLIER,
        },
        "moderate": {
            "min_confidence": 25,
            "max_position_size_usdt": 700,
            "max_daily_loss_percent": 20,
            "global_stop_loss_percent": 3,
            "base_size_percent": 0.05,
            "kelly_multiplier": 0.15,
            "sl_atr_multiple": 2.5,
            "tp_atr_multiple": 3.0,
            "max_sl_percent": 10,
            "buy_threshold": 15,
            "sell_threshold": -15,
            "strong_threshold": 30,
            "strategy_weights": {
                'trend_following': 0.25,
                'mean_reversion': 0.35,
                'breakout': 0.20,
                'ml_prediction': 0.20
            },
            "risk_multiplier": RISK_MULTIPLIER,
        },
        "aggressive": {
            "min_confidence": 20,
            "max_position_size_usdt": 500,
            "max_daily_loss_percent": 12,
            "global_stop_loss_percent": 4,
            "base_size_percent": 0.05,
            "kelly_multiplier": 0.25,
            "sl_atr_multiple": 1.2,
            "tp_atr_multiple": 3.0,
            "max_sl_percent": 6,
            "buy_threshold": 10,
            "sell_threshold": -10,
            "strong_threshold": 25,
            "strategy_weights": {
                'trend_following': 0.20,
                'mean_reversion': 0.40,
                'breakout': 0.25,
                'ml_prediction': 0.15
            },
            "risk_multiplier": RISK_MULTIPLIER,
        },
        "high": {
            "min_confidence": 15,
            "max_position_size_usdt": 800,
            "max_daily_loss_percent": 15,
            "global_stop_loss_percent": 5,
            "base_size_percent": 0.08,
            "kelly_multiplier": 0.35,
            "sl_atr_multiple": 1.0,
            "tp_atr_multiple": 4.0,
            "max_sl_percent": 8,
            "buy_threshold": 5,
            "sell_threshold": -5,
            "strong_threshold": 20,
            "strategy_weights": {
                'trend_following': 0.15,
                'mean_reversion': 0.45,
                'breakout': 0.30,
                'ml_prediction': 0.10
            },
            "risk_multiplier": RISK_MULTIPLIER,
        }
    }

    @classmethod
    def get_config(cls, aggressiveness: str):
        return cls.LEVELS.get(aggressiveness, cls.LEVELS["conservative"])