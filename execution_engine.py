import logging
import threading
import time
import numpy as np
import pandas as pd
from typing import Dict, List
from bybit_client import BybitClient
from advanced_risk_manager import AdvancedRiskManager
from config import SYMBOLS

class ExecutionEngine:
    def __init__(self, bybit_client: BybitClient, risk_manager: AdvancedRiskManager):
            self.client = bybit_client
            self.risk_manager = risk_manager
            # Use a dictionary for open orders: {order_id: {details}}
            self.open_orders: Dict[str, Dict] = {}
            # Internal position cache updated by WS: {symbol: {details}}
            self.position_cache: Dict[str, Dict] = {}
            self.trade_history = [] # Consider moving primary history to DB
            self.emergency_protocols = None # Must be set via set_emergency_protocols
            self.position_timeout = 30 # Timeout for verification
            self.max_retry_attempts = 3 # Max retries for placing orders
            self.execution_quality_log = []

            # --- Dependencies for advanced execution ---
            # Assuming these classes are defined elsewhere in the file or imported
            self.market_impact_model = MarketImpactModel()
            self.limit_order_strategies = LimitOrderStrategies()
            self.vwap_executor = VWAPExecutor(bybit_client)
            self.twap_executor = TWAPExecutor(bybit_client)
            self.smart_router = SmartOrderRouter(bybit_client)
            # --- End Dependencies ---

            self._state_lock = threading.Lock() # Lock for thread-safe updates to open_orders and position_cache
            self.logger = logging.getLogger('ExecutionEngine')

            # Initialize position cache via REST
            self._initialize_position_cache()
            
    def set_emergency_protocols(self, emergency_protocols):
        self.emergency_protocols = emergency_protocols

    def _initialize_position_cache(self):
            """Populates the initial position cache using REST API."""
            self.logger.info("Initializing position cache via REST...")
            try:
                # Use the client's method directly
                pos_response = self.client.get_position_info()
                if pos_response and pos_response.get('retCode') == 0:
                    with self._state_lock:
                        self.position_cache.clear() # Clear existing before refresh
                        count = 0
                        for pos in pos_response['result'].get('list', []):
                            symbol = pos.get('symbol')
                            size = float(pos.get('size', 0))
                            # Only cache if symbol exists and size > 0? Or cache all? Cache non-zero for relevance.
                            if symbol and size > 0:
                                self.position_cache[symbol] = {
                                    'size': size,
                                    'side': pos.get('side'),
                                    'avgPrice': float(pos.get('avgPrice', 0)),
                                    'positionValue': float(pos.get('positionValue', 0)),
                                    'unrealisedPnl': float(pos.get('unrealisedPnl', 0)),
                                    'liqPrice': float(pos.get('liqPrice', 0)),
                                    'updatedTime': int(pos.get('updatedTime', time.time()*1000)) # Store Bybit's timestamp
                                }
                                count += 1
                        self.logger.info(f"Initialized position cache with {count} open positions.")
                else:
                    err_msg = pos_response.get('retMsg', 'No response') if pos_response else 'No response'
                    self.logger.error(f"Failed to initialize position cache via REST: {err_msg}")
            except Exception as e:
                self.logger.error(f"Error initializing position cache: {e}", exc_info=True)

    def handle_private_ws_message(self, message: Dict):
            """Processes private WebSocket messages (orders, positions). MUST be called from WS thread."""
            topic = message.get("topic")
            data_list = message.get("data") # Data is usually a list

            if not topic or not isinstance(data_list, list):
                self.logger.warning(f"Received malformed private WS message: {message}")
                return

            with self._state_lock: # Ensure thread safety when modifying shared state
                try:
                    if topic == "order":
                        for order_data in data_list:
                            order_id = order_data.get('orderId')
                            order_status = order_data.get('orderStatus')
                            symbol = order_data.get('symbol')
                            self.logger.info(f"WS Order Update: {symbol} ID {order_id} -> {order_status}")

                            if order_id in self.open_orders:
                                # Update status in internal tracking
                                self.open_orders[order_id]['status'] = order_status
                                self.open_orders[order_id]['updatedTime'] = message.get('ts', time.time()*1000)
                                self.open_orders[order_id]['avgPrice'] = float(order_data.get('avgPrice', self.open_orders[order_id].get('avgPrice', 0)))
                                self.open_orders[order_id]['cumExecQty'] = float(order_data.get('cumExecQty', self.open_orders[order_id].get('cumExecQty', 0)))

                                # Remove order if it's in a final state
                                if order_status in ['Filled', 'Cancelled', 'Rejected', 'Expired', 'PartiallyFilledCanceled']:
                                    del self.open_orders[order_id]
                                    self.logger.info(f"Removed closed/failed order {order_id} from internal tracking.")
                            elif order_status not in ['Filled', 'Cancelled', 'Rejected', 'Expired', 'PartiallyFilledCanceled']:
                                # Log if we receive an update for an order not actively tracked but seems active
                                self.logger.warning(f"Received WS update for untracked active order: {order_id} ({symbol} {order_status})")

                    elif topic == "position":
                        for pos_data in data_list:
                            symbol = pos_data.get('symbol')
                            size = float(pos_data.get('size', 0))
                            side = pos_data.get('side')
                            avg_price = float(pos_data.get('avgPrice', 0))
                            pos_value = float(pos_data.get('positionValue', 0))
                            pnl = float(pos_data.get('unrealisedPnl', 0))
                            liq_price = float(pos_data.get('liqPrice', 0))
                            # Use Bybit's updatedTime if available, else WS message time
                            updated_time = int(pos_data.get('updatedTime', message.get('ts', time.time()*1000)))

                            if symbol:
                                self.logger.debug(f"WS Position Update: {symbol} Size={size} Side={side} PNL={pnl}")
                                # Update or create entry in cache
                                self.position_cache[symbol] = {
                                    'size': size,
                                    'side': side,
                                    'avgPrice': avg_price,
                                    'positionValue': pos_value,
                                    'unrealisedPnl': pnl,
                                    'liqPrice': liq_price,
                                    'updatedTime': updated_time # Store timestamp
                                }
                                # Don't delete if size is 0, just update it. Verification needs the 0 size info.
                                if size == 0:
                                    self.logger.info(f"Position size updated to 0 for {symbol} via WS.")

                except Exception as e:
                    self.logger.error(f"Error processing private WS message: {e} | Message: {message}", exc_info=True)
                    # Consider triggering reconciliation if errors persist

    def _validate_trade_decision(self, decision: Dict) -> bool:
            """Validates the structure and basic logic of a trade decision dictionary."""
            required_fields = ['symbol', 'action', 'quantity', 'position_size', 'current_price']
            for field in required_fields:
                if field not in decision:
                    self.logger.error(f"Trade validation failed: Missing field '{field}' in decision: {decision}")
                    return False
            # Check SYMBOLS from config
            if decision['symbol'] not in SYMBOLS: # Make sure SYMBOLS is accessible
                self.logger.error(f"Trade validation failed: Invalid symbol '{decision['symbol']}'")
                return False
            if decision['action'] not in ['BUY', 'SELL', 'HOLD']:
                self.logger.error(f"Trade validation failed: Invalid action '{decision['action']}'")
                return False
            # Allow HOLD action without quantity checks
            if decision['action'] == 'HOLD':
                return True
            # Checks for BUY/SELL actions
            if not isinstance(decision['quantity'], (int, float)) or decision['quantity'] <= 0:
                self.logger.error(f"Trade validation failed: Non-positive quantity '{decision['quantity']}'")
                return False
            if not isinstance(decision['position_size'], (int, float)) or decision['position_size'] <= 0:
                self.logger.error(f"Trade validation failed: Non-positive position_size '{decision['position_size']}'")
                return False
            if not isinstance(decision['current_price'], (int, float)) or decision['current_price'] <= 0:
                self.logger.error(f"Trade validation failed: Non-positive current_price '{decision['current_price']}'")
                return False

            # SL/TP checks only needed for BUY/SELL
            if 'stop_loss' not in decision or 'take_profit' not in decision:
                self.logger.error(f"Trade validation failed: Missing SL or TP for action '{decision['action']}'")
                return False
            if not isinstance(decision['stop_loss'], (int, float)) or decision['stop_loss'] <= 0:
                self.logger.error(f"Trade validation failed: Non-positive SL '{decision['stop_loss']}'")
                return False
            if not isinstance(decision['take_profit'], (int, float)) or decision['take_profit'] <= 0:
                self.logger.error(f"Trade validation failed: Non-positive TP '{decision['take_profit']}'")
                return False

            # Add checks for SL/TP placement relative to current_price
            price = decision['current_price']
            sl = decision['stop_loss']
            tp = decision['take_profit']
            if decision['action'] == 'BUY':
                if sl >= price:
                    self.logger.error(f"Trade validation failed: BUY SL ({sl}) >= Price ({price})")
                    return False
                if tp <= price:
                    self.logger.error(f"Trade validation failed: BUY TP ({tp}) <= Price ({price})")
                    return False
            elif decision['action'] == 'SELL':
                if sl <= price:
                    self.logger.error(f"Trade validation failed: SELL SL ({sl}) <= Price ({price})")
                    return False
                if tp >= price:
                    self.logger.error(f"Trade validation failed: SELL TP ({tp}) >= Price ({price})")
                    return False
            return True

    def execute_limit_order(self, decision: Dict, strategy: str = 'aggressive') -> Dict:
            """Executes a trade using a Limit Order based on smart routing."""
            try:
                # --- Validation and Pre-checks ---
                if not self._validate_trade_decision(decision):
                    return {'success': False, 'message': 'Trade validation failed'}

                symbol = decision['symbol']
                action = decision['action']

                if action == 'HOLD':
                    return {'success': True, 'message': 'No action needed'}

                if self.emergency_protocols and self.emergency_protocols.emergency_mode:
                    self.logger.warning(f"Skipping limit order for {symbol}: Emergency mode active.")
                    return {'success': False, 'message': 'Emergency mode active'}

                position_size_usdt = decision['position_size']
                quantity = decision['quantity']
                # Price at decision time, used for analysis/impact estimation
                current_price_at_decision = decision['current_price']

                # --- Risk Check ---
                risk_approval = self.risk_manager.can_trade(symbol, position_size_usdt)
                if not risk_approval.get('approved'):
                    reason = risk_approval.get('reason', 'No reason specified')
                    self.logger.warning(f"Limit order rejected by Risk Manager for {symbol}: {reason}")
                    return {'success': False, 'message': f'Risk management rejected trade: {reason}'}

                # --- Adjust Size if Necessary ---
                adjusted_size = risk_approval.get('adjusted_size', position_size_usdt)
                if adjusted_size != position_size_usdt:
                    if adjusted_size <= 0:
                        self.logger.warning(f"Risk adjustment resulted in non-positive size (${adjusted_size:.2f}) for {symbol}. Skipping limit order.")
                        return {'success': False, 'message': f'Risk adjustment resulted in non-positive size (${adjusted_size:.2f})'}
                    position_size_usdt = adjusted_size
                    if current_price_at_decision > 0:
                        quantity = position_size_usdt / current_price_at_decision
                        quantity = round(quantity, 3 if 'BTC' in symbol else 2 if 'ETH' in symbol else 1) # Example rounding
                        self.logger.info(f"Adjusted limit order size for {symbol} to ${position_size_usdt:.2f} ({quantity} units) due to risk rules: {risk_approval.get('reason', '')}")
                    else:
                        self.logger.error(f"Cannot calculate adjusted quantity for limit order {symbol}: decision price is zero.")
                        return {'success': False, 'message': 'Cannot calculate quantity with zero price after risk adjustment.'}

                # --- Final Quantity Check ---
                min_order_qty = 0.001 # Example
                if quantity < min_order_qty:
                    self.logger.warning(f"Skipping limit order for {symbol}: Final quantity {quantity} is below minimum {min_order_qty}.")
                    return {'success': False, 'message': f'Final quantity {quantity} is below minimum order size {min_order_qty}'}

                # --- Smart Routing / Price Calculation ---
                # Use current price at decision for routing analysis
                orderbook_analysis = self._analyze_order_book(symbol)
                optimal_execution = self.smart_router.find_optimal_execution(symbol, action, quantity, orderbook_analysis)

                side = 'Buy' if action == 'BUY' else 'Sell'

                # Decide execution based on router suggestion
                if optimal_execution['strategy'] == 'limit':
                    limit_price = optimal_execution['price']
                    # Ensure calculated limit price is valid
                    if limit_price <= 0:
                        self.logger.error(f"Smart router provided invalid limit price ({limit_price}) for {symbol}. Falling back to market.")
                        # Fallback to market order if limit price is invalid
                        decision['quantity'] = quantity # Update decision with potentially adjusted quantity
                        decision['position_size'] = position_size_usdt
                        return self.execute_enhanced_trade(decision)

                    market_impact = optimal_execution['estimated_impact']

                    self.logger.info(f"🎯 Attempting LIMIT {action} for {symbol} at ${limit_price:.4f}") # More precision for limit
                    self.logger.info(f"   📊 Quantity: {quantity:.4f} units (${position_size_usdt:.2f})")
                    self.logger.info(f"   🛡️ SL: ${decision['stop_loss']:.2f}, TP: ${decision['take_profit']:.2f}")
                    self.logger.info(f"   Router Strategy: {strategy}, Estimated Impact: {market_impact:.4f}%")

                    # --- Place Limit Order ---
                    order_response = self.client.place_order(
                        symbol=symbol,
                        side=side,
                        order_type='Limit',
                        qty=quantity,
                        price=limit_price,
                        stop_loss=decision['stop_loss'],
                        take_profit=decision['take_profit'],
                        time_in_force='GoodTillCancel' # Or 'PostOnly' etc. based on strategy
                    )

                    # --- Process Response ---
                    if order_response and order_response.get('retCode') == 0:
                        order_id = order_response['result']['orderId']

                        # --- Update Internal State ---
                        with self._state_lock:
                            self.open_orders[order_id] = {
                                'symbol': symbol, 'order_id': order_id, 'side': side,
                                'quantity': quantity, 'limit_price': limit_price,
                                'timestamp': time.time(), 'status': 'New', 'type': 'Limit',
                                'strategy': strategy
                            }
                        # --------------------------

                        # Log execution quality based on INTENDED price vs limit price placed
                        self._log_execution_quality('LIMIT_PLACED', symbol, current_price_at_decision, limit_price, quantity, market_impact)

                        # --- Prepare Trade Record (Order Placement) ---
                        trade_record = {**decision}
                        trade_record.update({
                            'timestamp': time.time(), 'side': side,
                            'entry_price': None, # Not filled yet
                            'limit_price_placed': limit_price,
                            'order_id': order_id,
                            'success': True, # Order placed successfully
                            'status': 'New' # Initial status
                        })
                        trade_record.pop('signals', None); trade_record.pop('market_context', None); trade_record.pop('trade_quality', None)
                        self.trade_history.append(trade_record)

                        self.logger.info(f"✅ Limit order for {symbol} placed successfully! Order ID: {order_id}")

                        # --- Return Success Result ---
                        return {
                            'success': True,
                            'order_id': order_id,
                            'symbol': symbol,
                            'side': side,
                            'quantity': quantity,
                            'limit_price': limit_price,
                            'strategy': strategy,
                            'execution_quality': optimal_execution['quality'],
                            'status': 'New' # Indicate order is placed but not filled
                        }
                    else:
                        # --- Handle Order Placement Failure ---
                        error_msg = order_response.get('retMsg', 'Unknown error') if order_response else 'No response'
                        ret_code = order_response.get('retCode', -1) if order_response else -1
                        self.logger.error(f"❌ Limit order placement failed for {symbol}: {error_msg} (Code: {ret_code})")
                        # Log failure
                        trade_record = {**decision}
                        trade_record.update({ 'timestamp': time.time(), 'side': side, 'success': False, 'error_message': f"Code {ret_code}: {error_msg}" })
                        trade_record.pop('signals', None); trade_record.pop('market_context', None); trade_record.pop('trade_quality', None)
                        self.trade_history.append(trade_record)
                        return {'success': False, 'message': f'Limit order failed ({ret_code}): {error_msg}'}
                else:
                    # --- Execute via Algorithmic Order (VWAP/TWAP) or Fallback Market ---
                    self.logger.info(f"Smart router suggests '{optimal_execution['strategy']}' for {symbol}. Executing...")
                    decision['quantity'] = quantity # Ensure decision uses potentially adjusted quantity
                    decision['position_size'] = position_size_usdt
                    return self._execute_algorithmic_order(decision, optimal_execution)

            except Exception as e:
                # --- Handle Unexpected Errors ---
                symbol_for_log = decision.get('symbol', 'UNKNOWN') if isinstance(decision, dict) else 'UNKNOWN'
                self.logger.error(f"❌ Exception during limit order execution for {symbol_for_log}: {e}", exc_info=True)
                # Log failure attempt if possible
                if isinstance(decision, dict) and 'action' in decision and decision['action'] != 'HOLD':
                    try:
                            trade_record = {**decision}
                            trade_record.update({ 'timestamp': time.time(), 'success': False, 'error_message': f'Limit Exec Exception: {e}'})
                            trade_record.pop('signals', None); trade_record.pop('market_context', None); trade_record.pop('trade_quality', None)
                            self.trade_history.append(trade_record)
                    except: pass
                return {'success': False, 'message': f'Limit order execution engine error: {e}'}

    def execute_adaptive_trade(self, decision: Dict) -> Dict:
            """Dynamically chooses execution strategy based on market conditions."""
            try:
                # --- Validation and Pre-checks ---
                if not self._validate_trade_decision(decision):
                    return {'success': False, 'message': 'Trade validation failed'}
                symbol = decision['symbol']
                action = decision['action']
                if action == 'HOLD': return {'success': True, 'message': 'No action needed'}
                if self.emergency_protocols and self.emergency_protocols.emergency_mode:
                    return {'success': False, 'message': 'Emergency mode active'}

                quantity = decision['quantity']
                current_price = decision['current_price'] # Price at decision time
                position_size_usdt = decision['position_size']

                # --- Risk Check ---
                risk_approval = self.risk_manager.can_trade(symbol, position_size_usdt)
                if not risk_approval.get('approved'):
                    return {'success': False, 'message': f'Risk rejected: {risk_approval.get("reason", "")}'}
                # Adjust size/qty if needed
                adjusted_size = risk_approval.get('adjusted_size', position_size_usdt)
                if adjusted_size != position_size_usdt:
                    if adjusted_size <= 0: return {'success': False, 'message': 'Adjusted size non-positive'}
                    position_size_usdt = adjusted_size
                    if current_price <= 0: return {'success': False, 'message': 'Cannot recalc qty, price zero'}
                    quantity = position_size_usdt / current_price
                    quantity = round(quantity, 3 if 'BTC' in symbol else 2) # Example rounding
                    self.logger.info(f"Adaptive trade size adjusted for {symbol} to ${position_size_usdt:.2f} ({quantity})")

                # Update decision dict with potentially adjusted values
                decision['position_size'] = position_size_usdt
                decision['quantity'] = quantity

                min_order_qty = 0.001 # Example
                if quantity < min_order_qty:
                    return {'success': False, 'message': f'Final quantity {quantity} below minimum'}


                # --- Adaptive Strategy Selection ---
                self.logger.info(f"🧠 Selecting adaptive strategy for {symbol} {action} {quantity:.4f}...")
                execution_quality = self._get_recent_execution_quality(symbol)
                market_conditions = self._analyze_market_conditions(symbol, quantity) # Needs implementation
                orderbook_analysis = self._analyze_order_book(symbol)

                strategy = 'smart_limit' # Default

                # --- Simplified Logic Example ---
                if market_conditions.get('high_volatility') or execution_quality.get('poor_fill_rate', False):
                    strategy = 'market'
                    self.logger.info("   -> High volatility or poor fills detected. Choosing MARKET.")
                elif quantity > market_conditions.get('optimal_limit_size', quantity * 1.5): # If order is large relative to near book depth
                    # Use VWAP/TWAP for larger orders
                    # More sophisticated logic could use orderbook depth analysis
                    vwap_threshold_usd = 10000 # Example threshold
                    if position_size_usdt > vwap_threshold_usd:
                        strategy = 'vwap'
                        self.logger.info(f"   -> Large order size (${position_size_usdt:.0f} > ${vwap_threshold_usd}). Choosing VWAP.")
                    else:
                        strategy = 'twap'
                        self.logger.info("   -> Moderate order size relative to depth. Choosing TWAP.")
                else:
                    strategy = 'smart_limit' # Default to smart limit if conditions seem ok
                    self.logger.info("   -> Conditions suitable. Choosing SMART LIMIT.")
                # --- End Simplified Logic ---

                # --- Execute Based on Selected Strategy ---
                if strategy == 'market':
                    return self.execute_enhanced_trade(decision) # Use enhanced market order
                elif strategy in ['vwap', 'twap']:
                    # Pass decision dict for SL/TP etc.
                    return self._execute_algorithmic_order(decision, {'strategy': strategy})
                else: # smart_limit or default limit
                    return self.execute_limit_order(decision, strategy='adaptive') # Pass specific strategy context

            except Exception as e:
                symbol_for_log = decision.get('symbol', 'UNKNOWN') if isinstance(decision, dict) else 'UNKNOWN'
                self.logger.error(f"❌ Exception during adaptive trade execution for {symbol_for_log}: {e}", exc_info=True)
                return {'success': False, 'message': f'Adaptive trade execution engine error: {e}'}

    def _execute_algorithmic_order(self, decision: Dict, execution_plan: Dict) -> Dict:
            """Routes execution to VWAP or TWAP executors."""
            symbol = decision['symbol']
            action = decision['action'] # BUY/SELL
            quantity = decision['quantity']
            # Current price not strictly needed by VWAP/TWAP, but good for context
            current_price = decision['current_price']
            strategy = execution_plan.get('strategy', 'unknown')

            self.logger.info(f"Routing to algorithmic execution: {strategy.upper()} for {symbol} {action} {quantity:.4f}")

            try:
                if strategy == 'vwap':
                    # Pass the full decision dictionary to VWAP/TWAP for SL/TP etc.
                    result = self.vwap_executor.execute_vwap(
                        symbol, action, quantity, decision, execution_plan.get('duration', 300) # Default 5 mins
                    )
                elif strategy == 'twap':
                    result = self.twap_executor.execute_twap(
                        symbol, action, quantity, decision, execution_plan.get('duration', 300) # Default 5 mins
                    )
                else:
                    self.logger.warning(f"Unknown algorithmic strategy '{strategy}' requested for {symbol}. Falling back to market.")
                    return self.execute_enhanced_trade(decision)

                # --- Log Algo Execution Attempt ---
                # Algo executors should return success status and details
                if result.get('success'):
                    # Log overall success, details might be logged within executor
                    self.logger.info(f"✅ {strategy.upper()} execution for {symbol} reported success. Total executed: {result.get('total_executed', 0):.4f}")
                    # Record a single trade entry representing the algo order
                    trade_record = {**decision}
                    trade_record.update({
                        'timestamp': time.time(),
                        'side': 'Buy' if action == 'BUY' else 'Sell',
                        'entry_price': result.get('average_price'), # Use avg price if available
                        'order_id': f"{strategy.upper()}_{int(time.time())}", # Placeholder ID
                        'success': True, # Request placed
                        'status': 'Completed' if result.get('completion_rate', 0) >= 0.99 else 'PartiallyCompleted', # Indicate algo execution status
                        'algo_strategy': strategy,
                        'total_executed_qty': result.get('total_executed')
                    })
                    trade_record.pop('signals', None); trade_record.pop('market_context', None); trade_record.pop('trade_quality', None)
                    self.trade_history.append(trade_record)
                else:
                    self.logger.error(f"❌ {strategy.upper()} execution for {symbol} failed: {result.get('message', 'No details')}")
                    trade_record = {**decision}
                    trade_record.update({ 'timestamp': time.time(), 'success': False, 'error_message': f"{strategy} Failed: {result.get('message', '')}" })
                    trade_record.pop('signals', None); trade_record.pop('market_context', None); trade_record.pop('trade_quality', None)
                    self.trade_history.append(trade_record)

                return result # Return the result from the algo executor

            except Exception as e:
                self.logger.error(f"❌ Exception routing to algorithmic execution ({strategy}) for {symbol}: {e}", exc_info=True)
                return {'success': False, 'message': f'Algorithmic execution routing error: {e}'}

    def _analyze_order_book(self, symbol: str, depth_levels: int = 10) -> Dict:
        try:
            orderbook = self.client.get_orderbook(symbol)
            if not orderbook or not orderbook.get('result'):
                return {}
            
            bids = orderbook['result']['b'][:depth_levels]
            asks = orderbook['result']['a'][:depth_levels]
            
            bid_prices = [float(bid[0]) for bid in bids]
            bid_sizes = [float(bid[1]) for bid in bids]
            ask_prices = [float(ask[0]) for ask in asks]
            ask_sizes = [float(ask[1]) for ask in asks]
            
            total_bid_volume = sum(bid_sizes)
            total_ask_volume = sum(ask_sizes)
            
            weighted_bid_price = sum(bid_prices[i] * bid_sizes[i] for i in range(len(bid_prices))) / total_bid_volume
            weighted_ask_price = sum(ask_prices[i] * ask_sizes[i] for i in range(len(ask_prices))) / total_ask_volume
            
            mid_price = (weighted_bid_price + weighted_ask_price) / 2
            spread = (weighted_ask_price - weighted_bid_price) / mid_price * 100
            
            depth_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            price_levels = []
            cumulative_bid = 0
            for i, (price, size) in enumerate(zip(bid_prices, bid_sizes)):
                cumulative_bid += size
                price_levels.append({
                    'side': 'bid',
                    'level': i + 1,
                    'price': price,
                    'size': size,
                    'cumulative': cumulative_bid
                })
            
            cumulative_ask = 0
            for i, (price, size) in enumerate(zip(ask_prices, ask_sizes)):
                cumulative_ask += size
                price_levels.append({
                    'side': 'ask',
                    'level': i + 1,
                    'price': price,
                    'size': size,
                    'cumulative': cumulative_ask
                })
            
            orderbook_strength = self._calculate_orderbook_strength(bid_sizes, ask_sizes)
            liquidity_density = self._calculate_liquidity_density(price_levels)
            
            return {
                'symbol': symbol,
                'timestamp': time.time(),
                'mid_price': mid_price,
                'weighted_bid': weighted_bid_price,
                'weighted_ask': weighted_ask_price,
                'spread_bps': spread * 100,
                'depth_imbalance': depth_imbalance,
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'price_levels': price_levels,
                'orderbook_strength': orderbook_strength,
                'liquidity_density': liquidity_density,
                'quality_score': self._calculate_orderbook_quality(spread, depth_imbalance, orderbook_strength)
            }
            
        except Exception as e:
            return {}

    def _calculate_orderbook_strength(self, bid_sizes: List[float], ask_sizes: List[float]) -> float:
        if not bid_sizes or not ask_sizes:
            return 0.5
        
        avg_bid_size = np.mean(bid_sizes)
        avg_ask_size = np.mean(ask_sizes)
        
        total_liquidity = sum(bid_sizes) + sum(ask_sizes)
        if total_liquidity == 0:
            return 0.5
        
        strength = (sum(bid_sizes) - sum(ask_sizes)) / total_liquidity
        return (strength + 1) / 2

    def _calculate_liquidity_density(self, price_levels: List[Dict]) -> float:
        if not price_levels:
            return 0.0
        
        total_size = sum(level['size'] for level in price_levels)
        if total_size == 0:
            return 0.0
        
        prices = [level['price'] for level in price_levels]
        price_range = max(prices) - min(prices)
        
        if price_range == 0:
            return 1.0
        
        density = total_size / price_range
        return min(density / 1000, 1.0)

    def _calculate_orderbook_quality(self, spread: float, imbalance: float, strength: float) -> float:
        spread_score = max(0, 1 - (spread / 0.1))
        imbalance_score = 1 - abs(imbalance)
        strength_score = strength
        
        quality = (spread_score * 0.4 + imbalance_score * 0.3 + strength_score * 0.3)
        return min(max(quality, 0), 1)

    def _log_execution_quality(self, order_type: str, symbol: str, intended_price: float, 
                             executed_price: float, quantity: float, market_impact: float):
        slippage = ((executed_price - intended_price) / intended_price) * 100
        if order_type == 'MARKET':
            slippage = abs(slippage)
        
        quality_metric = {
            'timestamp': time.time(),
            'symbol': symbol,
            'order_type': order_type,
            'intended_price': intended_price,
            'executed_price': executed_price,
            'quantity': quantity,
            'slippage': slippage,
            'market_impact': market_impact,
            'fill_quality': self._calculate_fill_quality(slippage, market_impact)
        }
        
        self.execution_quality_log.append(quality_metric)
        
        if len(self.execution_quality_log) > 1000:
            self.execution_quality_log = self.execution_quality_log[-1000:]

    def _calculate_fill_quality(self, slippage: float, market_impact: float) -> str:
        total_impact = abs(slippage) + market_impact
        if total_impact < 0.05:
            return 'EXCELLENT'
        elif total_impact < 0.1:
            return 'GOOD'
        elif total_impact < 0.2:
            return 'AVERAGE'
        else:
            return 'POOR'

    def _get_recent_execution_quality(self, symbol: str, lookback: int = 50) -> Dict:
        recent_trades = [t for t in self.execution_quality_log 
                        if t['symbol'] == symbol][-lookback:]
        
        if not recent_trades:
            return {'avg_slippage': 0, 'fill_rate': 1.0, 'poor_fill_rate': False}
        
        avg_slippage = sum(t['slippage'] for t in recent_trades) / len(recent_trades)
        poor_fills = len([t for t in recent_trades if t['fill_quality'] in ['AVERAGE', 'POOR']])
        poor_fill_rate = poor_fills / len(recent_trades)
        
        return {
            'avg_slippage': avg_slippage,
            'fill_rate': 1 - poor_fill_rate,
            'poor_fill_rate': poor_fill_rate > 0.3
        }

    def _analyze_market_conditions(self, symbol: str, quantity: float) -> Dict:
        try:
            orderbook = self.client.get_orderbook(symbol)
            if orderbook and orderbook.get('result'):
                bids = orderbook['result']['b']
                asks = orderbook['result']['a']
                
                bid_volume = sum(float(bid[1]) for bid in bids[:5])
                ask_volume = sum(float(ask[1]) for ask in asks[:5])
                
                spread = (float(asks[0][0]) - float(bids[0][0])) / float(bids[0][0]) * 100
                
                depth_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                
                optimal_limit_size = min(bid_volume, ask_volume) * 0.1
                
                return {
                    'spread': spread,
                    'depth_imbalance': depth_imbalance,
                    'high_volatility': spread > 0.1,
                    'optimal_limit_size': optimal_limit_size,
                    'liquidity_adequate': quantity < optimal_limit_size
                }
        except:
            pass
        
        return {
            'spread': 0.05,
            'depth_imbalance': 0,
            'high_volatility': False,
            'optimal_limit_size': quantity * 2,
            'liquidity_adequate': True
        }

    def monitor_and_adapt_execution(self):
        recent_quality = self._get_recent_execution_quality('ALL')
        
        if recent_quality.get('poor_fill_rate', False):
            print("🔄 Poor execution quality detected - adapting strategies")
        
        return recent_quality

    def execute_trade(self, decision: Dict):
        try:
            if not self._validate_trade_decision(decision):
                return {'success': False, 'message': 'Trade validation failed'}
            
            symbol = decision['symbol']
            action = decision['action']
            confidence = decision['confidence']
            current_price = decision['current_price']
            
            if action == 'HOLD':
                return {'success': True, 'message': 'No action needed'}
            
            if self.emergency_protocols and self.emergency_protocols.emergency_mode:
                return {'success': False, 'message': 'Emergency mode active - trading suspended'}
            
            position_size_usdt = self.risk_manager.calculate_position_size(confidence, symbol, current_price)
            
            risk_approval = self.risk_manager.can_trade(symbol, position_size_usdt)
            
            if not risk_approval['approved']:
                return {'success': False, 'message': f'Risk management rejected trade: {risk_approval["reason"]}'}
            
            if risk_approval['adjusted_size'] > 0:
                position_size_usdt = risk_approval['adjusted_size']
            
            quantity = position_size_usdt / current_price
            quantity = round(quantity, 3)
            
            print(f"Executing {action} for {symbol}: {quantity} units (${position_size_usdt:.2f})")
            
            if action == 'BUY':
                side = 'Buy'
                stop_loss = decision.get('stop_loss', current_price * 0.98)
                take_profit = decision.get('take_profit', current_price * 1.04)
            else:
                side = 'Sell' 
                stop_loss = decision.get('stop_loss', current_price * 1.02)
                take_profit = decision.get('take_profit', current_price * 0.96)
            
            market_impact = self.market_impact_model.estimate_impact(symbol, quantity, side)
            
            order_response = self.client.place_order(
                symbol=symbol,
                side=side,
                order_type='Market',
                qty=quantity,
                price=None,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if order_response and order_response.get('retCode') == 0:
                order_id = order_response['result']['orderId']
                executed_price = float(order_response['result']['avgPrice'])
                
                self.open_orders[symbol] = order_id
                
                self._log_execution_quality('MARKET', symbol, current_price, executed_price, quantity, market_impact)
                
                trade_record = {
                    'timestamp': time.time(),
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': executed_price,
                    'size_usdt': position_size_usdt,
                    'order_id': order_id,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                self.trade_history.append(trade_record)
                
                return {
                    'success': True, 
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
            else:
                error_msg = order_response.get('retMsg', 'Unknown error') if order_response else 'No response'
                return {'success': False, 'message': f'Order failed: {error_msg}'}
                
        except Exception as e:
            return {'success': False, 'message': f'Execution error: {e}'}
    
    def execute_enhanced_trade(self, decision: Dict):
            """Executes a trade based on the enhanced decision dictionary via Market Order."""
            try:
                # Re-validate just before execution
                if not self._validate_trade_decision(decision):
                    # Validation error already logged inside _validate_trade_decision
                    return {'success': False, 'message': 'Trade validation failed before execution'}

                symbol = decision['symbol']
                action = decision['action']

                if action == 'HOLD':
                    self.logger.info(f"HOLD decision for {symbol}, no execution needed.")
                    return {'success': True, 'message': 'No action needed'}

                # Check emergency status
                if self.emergency_protocols and self.emergency_protocols.emergency_mode:
                    self.logger.warning(f"Skipping trade for {symbol}: Emergency mode active.")
                    return {'success': False, 'message': 'Emergency mode active - trading suspended'}

                position_size_usdt = decision['position_size']
                quantity = decision['quantity']
                # Price used for risk checks/logging, actual execution is Market
                current_price_at_decision = decision['current_price']

                # --- Risk Check ---
                # Pass None for market_data initially, risk manager can fetch if needed
                risk_approval = self.risk_manager.can_trade(symbol, position_size_usdt, market_data=None)
                if not risk_approval.get('approved'):
                    reason = risk_approval.get('reason', 'No reason specified')
                    self.logger.warning(f"Trade rejected by Risk Manager for {symbol}: {reason}")
                    return {'success': False, 'message': f'Risk management rejected trade: {reason}'}

                # --- Adjust Size if Necessary ---
                adjusted_size = risk_approval.get('adjusted_size', position_size_usdt)
                if adjusted_size != position_size_usdt:
                    # Check if adjustment makes size/qty invalid
                    if adjusted_size <= 0:
                        self.logger.warning(f"Risk adjustment resulted in non-positive size (${adjusted_size:.2f}) for {symbol}. Skipping trade.")
                        return {'success': False, 'message': f'Risk adjustment resulted in non-positive size (${adjusted_size:.2f})'}

                    position_size_usdt = adjusted_size
                    if current_price_at_decision > 0:
                        # Recalculate quantity based on adjusted size and original decision price
                        quantity = position_size_usdt / current_price_at_decision
                        # Apply appropriate rounding based on symbol specs (e.g., 3 decimals for BTC)
                        # This needs refinement based on actual exchange minimums/precision
                        quantity = round(quantity, 3 if 'BTC' in symbol else 2 if 'ETH' in symbol else 1) # Example rounding
                        self.logger.info(f"Adjusted trade size for {symbol} to ${position_size_usdt:.2f} ({quantity} units) due to risk rules: {risk_approval.get('reason', '')}")
                    else:
                        self.logger.error(f"Cannot calculate adjusted quantity for {symbol}: decision price is zero.")
                        return {'success': False, 'message': 'Cannot calculate quantity with zero price after risk adjustment.'}

                # --- Final Quantity Check ---
                # Add check for minimum order size if known (needs exchange info)
                min_order_qty = 0.001 # Example for BTC
                if quantity < min_order_qty:
                    self.logger.warning(f"Skipping trade for {symbol}: Final quantity {quantity} is below minimum {min_order_qty}.")
                    return {'success': False, 'message': f'Final quantity {quantity} is below minimum order size {min_order_qty}'}

                # --- Log Intent ---
                self.logger.info(f"🎯 Attempting Enhanced {action} for {symbol}")
                self.logger.info(f"   📊 Quantity: {quantity:.4f} units (${position_size_usdt:.2f})")
                self.logger.info(f"   🛡️ Stop Loss: ${decision['stop_loss']:.2f}")
                self.logger.info(f"   🎯 Take Profit: ${decision['take_profit']:.2f}")
                self.logger.info(f"   ⚖️ Risk/Reward: {decision.get('risk_reward_ratio', 0):.2f}:1") # Use get with default
                self.logger.info(f"   📈 Confidence: {decision.get('confidence', 0):.1f}%")
                self.logger.info(f"   🌡️ Market Regime: {decision.get('market_regime', 'N/A')}")

                side = 'Buy' if action == 'BUY' else 'Sell'
                stop_loss = decision['stop_loss']
                take_profit = decision['take_profit']

                market_impact = self.market_impact_model.estimate_impact(symbol, quantity, side)

                # --- Place Order ---
                order_response = self.client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type='Market',
                    qty=quantity,
                    price=None, # Market order doesn't need price
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

                # --- Process Response ---
                if order_response and order_response.get('retCode') == 0:
                    order_id = order_response['result']['orderId']
                    # Market orders usually fill immediately, avgPrice might be '0' initially via REST.
                    # WS update will give the final price. Use decision price as temporary placeholder.
                    executed_price_str = order_response['result'].get('avgPrice', '0')
                    executed_price = float(executed_price_str) if executed_price_str and executed_price_str != '0' else current_price_at_decision

                    # --- Update Internal State ---
                    with self._state_lock:
                        self.open_orders[order_id] = {
                            'symbol': symbol,
                            'order_id': order_id,
                            'side': side,
                            'quantity': quantity,
                            'timestamp': time.time(),
                            'status': 'New', # Initial status, WS will update to Filled/etc.
                            'type': 'Market',
                            'avgPrice': executed_price # Store initial avgPrice
                        }
                    # --------------------------

                    self._log_execution_quality('MARKET', symbol, current_price_at_decision, executed_price, quantity, market_impact)

                    # --- Prepare Trade Record for DB/History ---
                    trade_record = {**decision} # Start with all decision data
                    trade_record.update({
                        'timestamp': time.time(), # Consider order creation timestamp from response if available
                        'side': side, # Use 'Buy'/'Sell'
                        'entry_price': executed_price, # Use best guess at executed price
                        'order_id': order_id,
                        'success': True # Mark trade record as successfully placed
                    })
                    # Remove potentially large objects before storing if needed
                    trade_record.pop('signals', None)
                    trade_record.pop('market_context', None)
                    trade_record.pop('trade_quality', None)
                    # Keep ml_prediction dict as it's small

                    self.trade_history.append(trade_record) # Append to local history

                    self.logger.info(f"✅ Enhanced trade for {symbol} placed successfully! Order ID: {order_id}")

                    # --- Return Success Result ---
                    return {
                        'success': True,
                        'order_id': order_id,
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'executed_price': executed_price, # Return best guess execution price
                        'position_size': position_size_usdt, # Return final executed size
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk_reward': decision.get('risk_reward_ratio', 0),
                        'confidence': decision.get('confidence', 0)
                    }
                else:
                    # --- Handle Order Placement Failure ---
                    error_msg = order_response.get('retMsg', 'Unknown error') if order_response else 'No response'
                    ret_code = order_response.get('retCode', -1) if order_response else -1
                    self.logger.error(f"❌ Enhanced trade order failed for {symbol}: {error_msg} (Code: {ret_code})")

                    # Log failed attempt to history/DB
                    trade_record = {**decision}
                    trade_record.update({
                        'timestamp': time.time(),
                        'side': side,
                        'entry_price': None,
                        'order_id': None,
                        'success': False,
                        'error_message': f"Code {ret_code}: {error_msg}"
                    })
                    trade_record.pop('signals', None)
                    trade_record.pop('market_context', None)
                    trade_record.pop('trade_quality', None)
                    self.trade_history.append(trade_record)

                    return {'success': False, 'message': f'Order failed ({ret_code}): {error_msg}'}

            except Exception as e:
                # --- Handle Unexpected Errors ---
                symbol_for_log = decision.get('symbol', 'UNKNOWN') if isinstance(decision, dict) else 'UNKNOWN'
                self.logger.error(f"❌ Exception during enhanced trade execution for {symbol_for_log}: {e}", exc_info=True)
                # Log failure attempt if possible
                if isinstance(decision, dict) and 'action' in decision and decision['action'] != 'HOLD':
                    try:
                        trade_record = {**decision}
                        trade_record.update({ 'timestamp': time.time(), 'success': False, 'error_message': f'Execution Exception: {e}'})
                        trade_record.pop('signals', None); trade_record.pop('market_context', None); trade_record.pop('trade_quality', None)
                        self.trade_history.append(trade_record)
                    except: pass # Avoid errors during error logging
                return {'success': False, 'message': f'Execution engine error: {e}'}
    
    def execute_trade_with_retry(self, decision: Dict, max_retries: int = None):
            """Attempts to execute a trade using adaptive strategy with retries on failure."""
            if max_retries is None:
                max_retries = self.max_retry_attempts

            for attempt in range(max_retries):
                try:
                    # Use adaptive execution by default for retries
                    self.logger.info(f"Trade attempt {attempt + 1}/{max_retries} for {decision.get('symbol')} using adaptive execution...")
                    result = self.execute_adaptive_trade(decision)

                    if result.get('success'):
                        # Check if order needs monitoring (e.g., Limit, Algo)
                        if 'order_id' in result and result.get('status') == 'New': # Check if it's a limit order status
                            self.logger.info(f"Order {result['order_id']} placed, requires monitoring.")
                        elif result.get('strategy') in ['VWAP', 'TWAP']:
                            self.logger.info(f"{result['strategy']} execution initiated.")
                        return result # Return success immediately
                    else:
                        self.logger.warning(f"⚠️ Trade attempt {attempt + 1} failed: {result.get('message', 'No details')}")
                        # Decide if error is retryable (e.g., timeout, rate limit vs insufficient funds)
                        msg_lower = result.get('message', '').lower()
                        is_retryable = "timeout" in msg_lower or \
                                    "rate limit" in msg_lower or \
                                    "busy" in msg_lower or \
                                    "connection" in msg_lower or \
                                    "order failed (10006)" in msg_lower # Example: Order queue full / system busy

                        # Do NOT retry critical errors like insufficient balance, invalid params etc.
                        is_critical_error = "balance" in msg_lower or \
                                            "margin" in msg_lower or \
                                            "parameter" in msg_lower or \
                                            "invalid" in msg_lower

                        if is_retryable and not is_critical_error and attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2 # Exponential backoff might be better
                            self.logger.info(f"🔄 Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            self.logger.error(f"❌ Trade failed after attempt {attempt + 1}. Not retrying (Retryable: {is_retryable}, Critical: {is_critical_error}).")
                            # Record final failure (already done within execute methods)
                            return result # Return the final failure result

                except Exception as e:
                    self.logger.error(f"❌ Exception during trade attempt {attempt + 1} for {decision.get('symbol')}: {e}", exc_info=True)
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        self.logger.info(f"🔄 Retrying after exception in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"❌ Trade failed after attempt {attempt + 1} due to exception.")
                        # Log failure attempt
                        try:
                            trade_record = {**decision}
                            trade_record.update({ 'timestamp': time.time(), 'success': False, 'error_message': f'Retry Exception: {e}'})
                            trade_record.pop('signals', None); trade_record.pop('market_context', None); trade_record.pop('trade_quality', None)
                            self.trade_history.append(trade_record)
                        except: pass
                        return {'success': False, 'message': f'All attempts failed after exception: {e}'}

            # Fallback if loop completes unexpectedly
            self.logger.error(f"❌ All {max_retries} trade attempts failed for {decision.get('symbol')}.")
            return {'success': False, 'message': f'All {max_retries} trade attempts failed'}
    
    def cancel_all_orders(self, symbol: str = None):
            """Cancels all open orders, optionally filtered by symbol."""
            try:
                params = {"category": "linear"}
                if symbol:
                    params["symbol"] = symbol
                # Assuming self.client._request exists and is correct for POST /v5/order/cancel-all
                response = self.client._request("POST", "/v5/order/cancel-all", params)

                if response and response.get('retCode') == 0:
                    self.logger.info(f"✅ Cancel all orders request successful for {symbol if symbol else 'all symbols'}")
                    # --- Update Internal State ---
                    with self._state_lock:
                        orders_to_remove = []
                        for order_id, order_details in self.open_orders.items():
                            if symbol is None or order_details.get('symbol') == symbol:
                                orders_to_remove.append(order_id)
                        removed_count = 0
                        for order_id in orders_to_remove:
                            if order_id in self.open_orders:
                                del self.open_orders[order_id]
                                removed_count += 1
                        self.logger.info(f"Removed {removed_count} orders from internal tracking after cancel-all for {symbol or 'all'}.")
                    # --------------------------
                else:
                    error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                    self.logger.error(f"❌ Cancel all orders request failed: {error_msg}")

                return response # Return raw response
            except Exception as e:
                self.logger.error(f"❌ Exception canceling orders: {e}", exc_info=True)
                return None
    
    def get_position_info(self, symbol: str = None):
            """Gets position info, preferring WS cache but falling back to REST."""
            self.logger.debug(f"Getting position info for {symbol or 'all'}...")
            with self._state_lock:
                if symbol:
                    cached = self.position_cache.get(symbol)
                    # Check if cache is reasonably fresh (e.g., < 30 seconds old)
                    is_fresh = cached and (time.time()*1000 - cached.get('updatedTime', 0) < 30000)
                    if cached and is_fresh:
                        self.logger.debug(f"Returning cached position for {symbol}")
                        # Format similar to REST response for compatibility
                        # Ensure all necessary fields from REST are included if possible
                        rest_like_cache = {
                            "symbol": symbol,
                            "size": str(cached.get('size', 0)), # REST uses strings
                            "side": cached.get('side'),
                            "avgPrice": str(cached.get('avgPrice', 0)),
                            "positionValue": str(cached.get('positionValue', 0)),
                            "unrealisedPnl": str(cached.get('unrealisedPnl', 0)),
                            "liqPrice": str(cached.get('liqPrice', 0)),
                            "updatedTime": str(int(cached.get('updatedTime', 0))), # REST uses string timestamp
                            # Add other fields if needed, e.g., leverage, markPrice (not usually in WS pos update)
                            "leverage": "0", # Placeholder
                            "markPrice": "0" # Placeholder
                        }
                        return {'retCode': 0, 'result': {'list': [rest_like_cache] if rest_like_cache['size'] != '0' else []}, 'retMsg': 'OK (from cache)'}
                # else: # Requesting all symbols - difficult to check freshness easily, rely on REST or reconcile
                #     pass

            # Fallback to REST API
            self.logger.debug(f"Cache miss or stale for {symbol or 'all'}, fetching via REST...")
            try:
                response = self.client.get_position_info(symbol) # Use client's method
                # Optionally update cache here if REST was used? Handled by reconcile.
                if response and response.get('retCode') == 0:
                    # Update cache upon successful REST fetch for consistency
                    with self._state_lock:
                        if symbol:
                            pos_list = response['result'].get('list', [])
                            pos_data = pos_list[0] if pos_list else None
                            if pos_data:
                                self.position_cache[symbol] = {
                                    'size': float(pos_data.get('size', 0)), 'side': pos_data.get('side'),
                                    'avgPrice': float(pos_data.get('avgPrice', 0)), 'positionValue': float(pos_data.get('positionValue', 0)),
                                    'unrealisedPnl': float(pos_data.get('unrealisedPnl', 0)), 'liqPrice': float(pos_data.get('liqPrice', 0)),
                                    'updatedTime': int(pos_data.get('updatedTime', time.time()*1000))
                                }
                            elif symbol in self.position_cache: # If REST shows no position, clear cache
                                del self.position_cache[symbol]
                        else: # Update all symbols
                            self.position_cache.clear()
                            for pos_data in response['result'].get('list', []):
                                    sym = pos_data.get('symbol')
                                    if sym and float(pos_data.get('size', 0)) > 0:
                                        self.position_cache[sym] = {
                                            'size': float(pos_data.get('size', 0)), 'side': pos_data.get('side'),
                                            'avgPrice': float(pos_data.get('avgPrice', 0)), 'positionValue': float(pos_data.get('positionValue', 0)),
                                            'unrealisedPnl': float(pos_data.get('unrealisedPnl', 0)), 'liqPrice': float(pos_data.get('liqPrice', 0)),
                                            'updatedTime': int(pos_data.get('updatedTime', time.time()*1000))
                                        }
                return response
            except Exception as e:
                self.logger.error(f"❌ Error getting position info via REST fallback for {symbol or 'all'}: {e}")
                # Ensure error handler is called if defined in BybitClient's method
                # If not, call it here:
                # if self.client.error_handler:
                #     self.client.error_handler.handle_api_error(e, f"get_position_info ({symbol or 'all'})")
                return None # Or return error structure
    
    def get_order_status(self, symbol: str, order_id: str):
            """Gets order status, checking internal state first."""
            with self._state_lock:
                if order_id in self.open_orders:
                    order_details = self.open_orders[order_id].copy() # Get a copy
                    # Check if recently updated by WS (e.g., within 30s)
                    ws_update_time = order_details.get('updatedTime', 0)
                    is_fresh = time.time()*1000 - ws_update_time < 30000
                    current_status = order_details.get('status')

                    # If WS provided a final status recently, trust it short-term
                    if is_fresh and current_status in ['Filled', 'Cancelled', 'Rejected', 'Expired', 'PartiallyFilledCanceled']:
                        self.logger.debug(f"Returning cached final order status for {order_id}: {current_status}")
                        # Format similar to REST /v5/order/history response
                        rest_like_cache = {
                            "symbol": symbol, "orderId": order_id, "side": order_details.get('side'),
                            "qty": str(order_details.get('quantity', 0)), "orderStatus": current_status,
                            "avgPrice": str(order_details.get('avgPrice', 0)),
                            "cumExecQty": str(order_details.get('cumExecQty', 0)),
                            "orderType": order_details.get('type', 'Unknown'),
                            "createTime": str(int(order_details.get('timestamp', 0)*1000)), # Approx
                            "updateTime": str(int(ws_update_time)),
                            # Add other fields if available and needed
                        }
                        return {'retCode': 0, 'result': {'list': [rest_like_cache]}, 'retMsg': 'OK (from cache)'}
                    elif is_fresh:
                        self.logger.debug(f"Returning cached active order status for {order_id}: {current_status}")
                        # Still return cached non-final status if fresh
                        rest_like_cache = { "symbol": symbol, "orderId": order_id, "orderStatus": current_status, #... other fields ...
                                            }
                        return {'retCode': 0, 'result': {'list': [rest_like_cache]}, 'retMsg': 'OK (from cache)'}


            # Fallback to REST API if not found, stale, or still 'New'/'Unknown'
            self.logger.debug(f"Cache miss, stale, or non-final status for order {order_id}. Fetching via REST...")
            try:
                # Assuming self.client.get_order_status exists and wraps the API call to /v5/order/history
                response = self.client.get_order_status(symbol, order_id)

                # Update internal cache based on REST result
                if response and response.get('retCode') == 0 and response['result'].get('list'):
                    rest_data = response['result']['list'][0]
                    rest_status = rest_data.get('orderStatus')
                    rest_order_id = rest_data.get('orderId')
                    with self._state_lock:
                        if rest_order_id in self.open_orders:
                            self.open_orders[rest_order_id]['status'] = rest_status
                            self.open_orders[rest_order_id]['avgPrice'] = float(rest_data.get('avgPrice', 0))
                            self.open_orders[rest_order_id]['cumExecQty'] = float(rest_data.get('cumExecQty', 0))
                            self.open_orders[rest_order_id]['updatedTime'] = int(rest_data.get('updateTime', time.time()*1000))
                            if rest_status in ['Filled', 'Cancelled', 'Rejected', 'Expired', 'PartiallyFilledCanceled']:
                                    del self.open_orders[rest_order_id]
                                    self.logger.info(f"Removed order {rest_order_id} from tracking based on REST status: {rest_status}")
                        elif rest_status not in ['Filled', 'Cancelled', 'Rejected', 'Expired', 'PartiallyFilledCanceled']:
                                # If REST shows an active order not tracked, maybe add it back? Log for now.
                                self.logger.warning(f"REST found active order {rest_order_id} ({rest_status}) not currently tracked internally.")

                return response
            except Exception as e:
                self.logger.error(f"❌ Error getting order status via REST fallback for {order_id}: {e}")
                # Ensure error handler is called if needed (might be in client method)
                return None

    def execute_guaranteed_position_closure(self, symbol: str, max_attempts: int = 3) -> Dict:
            """Attempts closure using multiple methods until verified."""
            self.logger.info(f"🛡️ Attempting guaranteed position closure for {symbol}")
            closure_methods = [self.close_position, self._force_close_position, self._emergency_position_closure]
            final_result = {'success': False, 'message': 'Closure attempts failed'}

            for attempt in range(max_attempts):
                # Cycle through methods: close -> force_close -> emergency -> close ...
                method_to_try = closure_methods[attempt % len(closure_methods)]
                self.logger.info(f"Guaranteed closure attempt {attempt + 1}/{max_attempts} for {symbol} using {method_to_try.__name__}...")
                try:
                    # Execute the closure method
                    result = method_to_try(symbol)
                    # Store the result of the *last* attempt regardless of verification success
                    final_result = result

                    # Check if position is actually closed using enhanced verification
                    # Increase timeout on later attempts
                    if self._verify_position_closure(symbol, timeout=7 + attempt * 3):
                        self.logger.info(f"✅ Guaranteed closure for {symbol} successful and VERIFIED after attempt {attempt + 1}.")
                        # Ensure final result reflects success
                        final_result['success'] = True
                        final_result['message'] = final_result.get('message', 'Position closed and verified.')
                        return final_result # Exit loop on verified success
                    else:
                        self.logger.warning(f"Closure attempt {attempt + 1} for {symbol} finished (Method Success: {result.get('success', False)}), but verification FAILED. Retrying...")
                        # Add delay before next attempt
                        time.sleep(1 + attempt)

                except Exception as e:
                    self.logger.error(f"❌ Exception in guaranteed closure attempt {attempt + 1} ({method_to_try.__name__}) for {symbol}: {e}", exc_info=True)
                    final_result = {'success': False, 'message': f'Exception during closure attempt {attempt + 1}: {e}'}
                    # Add delay before next attempt
                    time.sleep(2 + attempt)

            self.logger.error(f"❌ All {max_attempts} guaranteed position closure attempts failed or could not be verified for {symbol}. Final attempt result: {final_result}")
            return final_result

    def _force_close_position(self, symbol: str) -> Dict:
            """Attempts to close position aggressively with market order, retrying smaller qty."""
            # (Implementation provided previously is mostly correct, ensure logging and verification call)
            self.logger.warning(f"Attempting FORCE CLOSE for {symbol} using market orders...")
            max_internal_retries = 2
            initial_quantity = 0
            close_side = ''

            try:
                # Get initial quantity from REST for safety
                position_response = self.client.get_position_info(symbol)
                if not position_response or position_response.get('retCode') != 0:
                    return {'success': False, 'message': 'Failed to get position info for force close'}
                positions = position_response['result'].get('list', [])
                position = next((p for p in positions if p['symbol'] == symbol), None)

                if not position or float(position.get('size', 0)) == 0:
                    self.logger.info(f"No position found for {symbol} during force close attempt.")
                    return {'success': True, 'message': 'No position found to force close'}

                initial_quantity = float(position['size'])
                current_side = position['side']
                close_side = 'Buy' if current_side == 'Sell' else 'Sell'

            except Exception as e:
                self.logger.error(f"Error getting position info in _force_close_position: {e}")
                return {'success': False, 'message': f'Error getting position info: {e}'}

            quantity_to_close = initial_quantity
            for i in range(max_internal_retries):
                self.logger.info(f"Force close attempt {i + 1}: Closing {quantity_to_close:.4f} {symbol} via {close_side} market order.")
                try:
                    order_response = self.client.place_order(
                        symbol=symbol, side=close_side, order_type='Market',
                        qty=quantity_to_close, price=None, stop_loss=None, take_profit=None
                    )

                    if order_response and order_response.get('retCode') == 0:
                        order_id = order_response['result']['orderId']
                        self.logger.info(f"✅ Force close market order placed successfully. Order ID: {order_id}")
                        # Don't rely solely on placement, verify closure
                        if self._verify_position_closure(symbol, timeout=10): # Use verification
                            return { 'success': True, 'order_id': order_id, 'method': 'force_close' }
                        else:
                            self.logger.warning(f"Force close order placed for {symbol}, but verification failed. Checking remaining size...")
                            # Check remaining size via REST again before next attempt
                            remaining_pos = self.client.get_position_info(symbol)
                            if remaining_pos and remaining_pos.get('retCode') == 0:
                                rem_list = remaining_pos['result'].get('list', [])
                                rem_p = next((p for p in rem_list if p['symbol'] == symbol), None)
                                if rem_p and float(rem_p.get('size', 0)) > 0:
                                    quantity_to_close = float(rem_p['size']) # Update qty for next attempt
                                    self.logger.info(f"Remaining size: {quantity_to_close}. Retrying force close.")
                                else: # Position IS closed now
                                    self.logger.info("Position closed after REST check during force close retry.")
                                    return { 'success': True, 'order_id': order_id, 'method': 'force_close_verified_late' }
                            else:
                                self.logger.error("Could not get remaining position size during force close retry.")
                                break # Break and return failure if can't get remaining size
                    else:
                        error_msg = order_response.get('retMsg', 'Unknown error') if order_response else 'No response'
                        self.logger.error(f"❌ Force close market order attempt {i + 1} failed: {error_msg}")
                        time.sleep(1 + i) # Wait longer

                except Exception as e:
                    self.logger.error(f"❌ Exception during force close attempt {i + 1}: {e}")
                    time.sleep(2 + i)

            self.logger.error(f"❌ Force close failed for {symbol} after {max_internal_retries} attempts.")
            return {'success': False, 'message': 'Force close failed after all retries'}

    def _emergency_position_closure(self, symbol: str) -> Dict:
            """Try alternative closure methods if force close fails."""
            # (Implementation provided previously is mostly correct, ensure logging and verification call)
            self.logger.critical(f"🚨 Executing EMERGENCY closure methods for {symbol}...")
            methods = [
                self._try_reduce_and_close, # Try multiple smaller market orders
                self._try_multiple_small_orders, # Try multiple small IOC limit orders
                self._try_different_order_types # Cycle Market/Limit IOC
            ]
            final_result = {'success': False, 'message': 'All emergency closure methods failed'}

            for method in methods:
                self.logger.info(f"Emergency closure: Trying method '{method.__name__}'...")
                try:
                    result = method(symbol)
                    final_result = result # Store last result
                    # Check verification AFTER the method attempt
                    if self._verify_position_closure(symbol, timeout=10):
                        self.logger.info(f"✅ Emergency closure for {symbol} succeeded and verified using {method.__name__}.")
                        # Ensure result reflects verified success
                        result['success'] = True
                        return result
                    else:
                        self.logger.warning(f"Emergency method {method.__name__} finished (Reported Success: {result.get('success', False)}), but verification failed. Trying next method...")
                        # Only continue if the method itself didn't report definite success AND verification failed
                        if result.get('success'): # If method thought it worked but verify failed, log issue but continue
                            self.logger.error(f"Potential issue: {method.__name__} reported success but position not verified closed.")

                    time.sleep(1) # Small delay between methods
                except Exception as e:
                    self.logger.error(f"Exception during emergency method {method.__name__}: {e}")
                    final_result = {'success': False, 'message': f'Exception in {method.__name__}: {e}'}
                    time.sleep(2)

            self.logger.error(f"❌ All emergency closure methods failed or could not be verified for {symbol}. Final result: {final_result}")
            return final_result

    def _try_reduce_and_close(self, symbol: str) -> Dict:
        try:
            position_response = self.client.get_position_info(symbol)
            if not position_response or position_response.get('retCode') != 0:
                return {'success': False}
            
            positions = position_response['result']['list']
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position or float(position['size']) == 0:
                return {'success': True, 'method': 'reduce_and_close', 'message': 'Position already closed'}
            
            quantity = float(position['size'])
            close_side = 'Buy' if position['side'] == 'Sell' else 'Sell'
            
            num_chunks = 3
            chunk_size = quantity / num_chunks
            
            for i in range(num_chunks):
                try:
                    order_response = self.client.place_order(
                        symbol=symbol,
                        side=close_side,
                        order_type='Market',
                        qty=chunk_size,
                        price=None
                    )
                    
                    if order_response and order_response.get('retCode') == 0:
                        print(f"✅ Closed chunk {i + 1} of {symbol}")
                        time.sleep(0.5)
                    
                except Exception as e:
                    print(f"❌ Failed to close chunk {i + 1}: {e}")
                    continue
            
            if self._verify_position_closure(symbol):
                if symbol in self.open_orders:
                    del self.open_orders[symbol]
                return {'success': True, 'method': 'reduce_and_close'}
            else:
                return {'success': False}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _try_multiple_small_orders(self, symbol: str) -> Dict:
        try:
            position_response = self.client.get_position_info(symbol)
            if not position_response or position_response.get('retCode') != 0:
                return {'success': False}
            
            positions = position_response['result']['list']
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position or float(position['size']) == 0:
                return {'success': True, 'method': 'multiple_small', 'message': 'Position already closed'}
            
            quantity = float(position['size'])
            close_side = 'Buy' if position['side'] == 'Sell' else 'Sell'
            
            order_sizes = self._calculate_optimal_chunk_sizes(quantity, symbol)
            
            for i, chunk_size in enumerate(order_sizes):
                try:
                    order_response = self.client.place_order(
                        symbol=symbol,
                        side=close_side,
                        order_type='Limit',
                        qty=chunk_size,
                        price=self._get_aggressive_price(symbol, close_side),
                        time_in_force='ImmediateOrCancel'
                    )
                    
                    if order_response and order_response.get('retCode') == 0:
                        print(f"✅ Small order {i + 1} placed for {symbol}")
                    
                    time.sleep(0.2)
                    
                except Exception as e:
                    print(f"❌ Failed to place small order {i + 1}: {e}")
                    continue
            
            remaining_response = self.client.get_position_info(symbol)
            if remaining_response and remaining_response.get('retCode') == 0:
                remaining_positions = remaining_response['result']['list']
                remaining_position = next((p for p in remaining_positions if p['symbol'] == symbol), None)
                
                if remaining_position and float(remaining_position['size']) > 0:
                    remaining_quantity = float(remaining_position['size'])
                    market_response = self.client.place_order(
                        symbol=symbol,
                        side=close_side,
                        order_type='Market',
                        qty=remaining_quantity,
                        price=None
                    )
                    
                    if market_response and market_response.get('retCode') == 0:
                        print(f"✅ Closed remaining {remaining_quantity} via market order")
            
            if self._verify_position_closure(symbol):
                if symbol in self.open_orders:
                    del self.open_orders[symbol]
                return {'success': True, 'method': 'multiple_small'}
            else:
                return {'success': False}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _try_different_order_types(self, symbol: str) -> Dict:
        try:
            position_response = self.client.get_position_info(symbol)
            if not position_response or position_response.get('retCode') != 0:
                return {'success': False}
            
            positions = position_response['result']['list']
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position or float(position['size']) == 0:
                return {'success': True, 'method': 'different_types', 'message': 'Position already closed'}
            
            quantity = float(position['size'])
            close_side = 'Buy' if position['side'] == 'Sell' else 'Sell'
            
            order_types = ['Market', 'Limit', 'Market']
            prices = [None, self._get_aggressive_price(symbol, close_side), None]
            
            for i, (order_type, price) in enumerate(zip(order_types, prices)):
                try:
                    order_response = self.client.place_order(
                        symbol=symbol,
                        side=close_side,
                        order_type=order_type,
                        qty=quantity,
                        price=price,
                        time_in_force='ImmediateOrCancel' if order_type == 'Limit' else 'GoodTillCancel'
                    )
                    
                    if order_response and order_response.get('retCode') == 0:
                        print(f"✅ {order_type} order executed for {symbol}")
                        
                        if self._verify_position_closure(symbol):
                            if symbol in self.open_orders:
                                del self.open_orders[symbol]
                            return {'success': True, 'method': 'different_types', 'order_type': order_type}
                    
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"❌ {order_type} order failed: {e}")
                    continue
            
            return {'success': False}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _calculate_optimal_chunk_sizes(self, total_quantity: float, symbol: str) -> List[float]:
        orderbook = self._analyze_order_book(symbol)
        if not orderbook:
            return [total_quantity * 0.5, total_quantity * 0.5]
        
        avg_level_size = np.mean([level['size'] for level in orderbook['price_levels'][:5]])
        optimal_chunk = min(total_quantity * 0.3, avg_level_size * 0.8)
        
        num_chunks = max(2, min(5, int(total_quantity / optimal_chunk)))
        base_chunk = total_quantity / num_chunks
        
        chunks = [base_chunk] * num_chunks
        chunks[-1] = total_quantity - sum(chunks[:-1])
        
        return [max(0.001, chunk) for chunk in chunks]

    def _get_aggressive_price(self, symbol: str, side: str) -> float:
        orderbook = self._analyze_order_book(symbol)
        if not orderbook:
            return 0.0
        
        if side == 'Buy':
            return orderbook['weighted_ask'] * 0.999
        else:
            return orderbook['weighted_bid'] * 1.001

    def _verify_position_closure(self, symbol: str, timeout: int = 10) -> bool:
            """Verify position closure, checking WS cache first, then REST with timeout."""
            start_time = time.time()
            self.logger.info(f"Verifying position closure for {symbol} (timeout {timeout}s)...")

            while time.time() - start_time < timeout:
                try:
                    # 1. Check WebSocket cache
                    with self._state_lock:
                        cached_pos = self.position_cache.get(symbol)
                        # Check if size is exactly 0 and if the update is reasonably recent
                        if cached_pos and cached_pos.get('size') == 0:
                            # Use a slightly longer window for verification check than general use
                            if time.time()*1000 - cached_pos.get('updatedTime', 0) < 20000: # 20 seconds
                                self.logger.info(f"✅ Verified position closure for {symbol} via WS cache.")
                                return True

                    # 2. If cache doesn't confirm or is old/missing, check REST API
                    self.logger.debug(f"WS cache doesn't confirm closure for {symbol}, checking REST...")
                    position_response = self.client.get_position_info(symbol)
                    if position_response and position_response.get('retCode') == 0:
                        positions = position_response['result'].get('list', [])
                        position = next((p for p in positions if p['symbol'] == symbol), None)

                        # Check if position list is empty OR the specific symbol has size 0
                        if not position or float(position.get('size', 0)) == 0:
                            self.logger.info(f"✅ Verified position closure for {symbol} via REST API.")
                            # Update cache based on REST confirmation
                            with self._state_lock:
                                if symbol in self.position_cache:
                                    # Update existing entry
                                    self.position_cache[symbol]['size'] = 0
                                    self.position_cache[symbol]['updatedTime'] = int(position.get('updatedTime', time.time()*1000)) if position else time.time()*1000
                                # else: # If it wasn't in cache, no need to add a zero entry
                            return True

                    # Wait before next check
                    time.sleep(1.5) # Check REST less frequently

                except Exception as e:
                    self.logger.error(f"❌ Error during position closure verification for {symbol}: {e}")
                    time.sleep(2) # Longer sleep on error

            self.logger.warning(f"❌ Position closure verification timed out for {symbol} after {timeout}s.")
            return False

    def validate_order_execution(self, symbol: str, order_id: str, expected_side: str, expected_quantity: float) -> Dict:
            """Validate order execution, checking WS state first, then confirming via REST."""
            validation_result = {'valid': False, 'reason': '', 'checked_via': 'none'}
            self.logger.debug(f"Validating order execution for {order_id} ({symbol})...")

            # 1. Check internal state (updated by WS)
            order_state = None
            with self._state_lock:
                order_state = self.open_orders.get(order_id)

            if order_state:
                ws_status = order_state.get('status')
                ws_cum_qty = order_state.get('cumExecQty', 0)
                ws_avg_price = order_state.get('avgPrice', 0)
                ws_timestamp = order_state.get('updatedTime', 0)
                is_fresh = time.time()*1000 - ws_timestamp < 30000 # 30s freshness

                # If WS shows a final state recently, use it for initial assessment
                if is_fresh and ws_status in ['Filled', 'PartiallyFilled', 'Cancelled', 'Rejected', 'Expired', 'PartiallyFilledCanceled']:
                    validation_result['reason'] = f"WS reports final status: {ws_status}"
                    validation_result['checked_via'] = 'ws_state_final'
                    # Proceed to REST check for full confirmation details
                elif is_fresh:
                    validation_result['reason'] = f"WS reports active status: {ws_status}"
                    validation_result['checked_via'] = 'ws_state_active'
                    # Proceed to REST check
                else:
                    validation_result['reason'] = f"WS state is stale or status is 'New'"
                    validation_result['checked_via'] = 'ws_state_stale'
                    # Proceed to REST check
            else:
                validation_result['reason'] = f"Order {order_id} not found in internal tracking (potentially filled/cancelled)."
                validation_result['checked_via'] = 'ws_state_closed'
                # Proceed to REST check for definite confirmation

            # 2. Fallback/Confirmation: Check REST API
            try:
                self.logger.debug(f"Validating order {order_id} via REST API for confirmation...")
                # Assuming self.client.get_order_status exists and calls /v5/order/history
                order_status_resp = self.client.get_order_status(symbol, order_id)

                if not order_status_resp or order_status_resp.get('retCode') != 0:
                    rest_reason = f"Could not fetch order status via REST (retCode: {order_status_resp.get('retCode', 'N/A')})"
                    validation_result.update({'valid': False, 'reason': f"{validation_result['reason']}. REST Check: {rest_reason}", 'checked_via': 'rest_failed'})
                    return validation_result

                order_list = order_status_resp['result'].get('list', [])
                order_info = order_list[0] if order_list else None

                if not order_info:
                    validation_result.update({'valid': False, 'reason': f"{validation_result['reason']}. REST Check: Order not found", 'checked_via': 'rest_not_found'})
                    # If WS thought it was closed AND REST confirms not found, it's likely validly closed/never existed
                    if validation_result['checked_via'] == 'ws_state_closed':
                        validation_result['valid'] = True # Assume cancellation/rejection happened correctly
                        validation_result['reason'] = "Order closed/cancelled (confirmed by WS state + REST not found)"
                    return validation_result

                # --- Extract REST Data ---
                actual_side = order_info.get('side') # 'Buy' or 'Sell'
                actual_quantity = float(order_info.get('qty', 0)) # Requested quantity
                order_status = order_info.get('orderStatus') # e.g., 'Filled', 'PartiallyFilled', 'Cancelled'
                avg_price = float(order_info.get('avgPrice', 0))
                cum_exec_qty = float(order_info.get('cumExecQty', 0))

                validation_result.update({
                    'order_status': order_status,
                    'side_match': actual_side == expected_side,
                    'quantity_request_match': abs(actual_quantity - expected_quantity) < 0.00001, # Check requested quantity
                    'partially_filled': order_status == 'PartiallyFilled',
                    'fully_filled': order_status == 'Filled',
                    'executed_qty': cum_exec_qty,
                    'avg_price': avg_price,
                    'checked_via': validation_result['checked_via'] + '+rest_success'
                })

                # Define 'valid' based on whether it's correctly filled (fully or partially)
                is_executed_or_partially = order_status in ['Filled', 'PartiallyFilled']
                validation_result['valid'] = (
                    validation_result['side_match'] and
                    is_executed_or_partially and
                    cum_exec_qty > 0 # Ensure *some* quantity was actually executed
                    # We don't check quantity_request_match here, as partial fills are valid executions
                )

                # Update reason if validation failed based on REST
                if not validation_result['valid']:
                    mismatches = []
                    if not validation_result['side_match']: mismatches.append("side")
                    if not is_executed_or_partially: mismatches.append("not_executed")
                    # If partially filled but exec_qty is 0, that's an issue
                    if order_status == 'PartiallyFilled' and cum_exec_qty <= 0 : mismatches.append("partial_zero_exec_qty")
                    # If filled but exec_qty is 0, that's an issue
                    if order_status == 'Filled' and cum_exec_qty <= 0: mismatches.append("filled_zero_exec_qty")

                    validation_result['reason'] = f"REST validation failed: Mismatches={','.join(mismatches)}, Status={order_status}, ExecQty={cum_exec_qty}"
                else:
                    validation_result['reason'] = f"REST validation passed: Status={order_status}, ExecQty={cum_exec_qty}"


                return validation_result

            except Exception as e:
                self.logger.error(f"Error during REST order validation for {order_id}: {e}", exc_info=True)
                validation_result.update({'valid': False, 'reason': f"{validation_result['reason']}. REST Check Error: {e}", 'checked_via': validation_result['checked_via'] + '+rest_exception'})
                return validation_result

    def reconcile_positions(self, perform_rest_check: bool = True) -> Dict:
            """Reconcile internal WS position cache with exchange state via REST API."""
            reconciliation = {
                'timestamp': time.time(),
                'rest_check_performed': False,
                'ws_cache_count': 0,
                'rest_count': 0,
                'matched': [],
                'mismatched_details': [], # Differences between WS cache and REST
                'unexpected_on_rest': [], # Found on REST, not in WS cache
                'missing_on_rest': [], # Found in WS cache, missing on REST
                'errors': []
            }
            rest_fetch_successful = False
            actual_positions_rest = {}
            ws_cache_snapshot = {}

            # --- Get WS Cache State ---
            try:
                with self._state_lock:
                    # Take a snapshot for comparison
                    ws_cache_snapshot = {sym: data.copy() for sym, data in self.position_cache.items()}
                reconciliation['ws_cache_count'] = len(ws_cache_snapshot)
            except Exception as e:
                self.logger.error(f"Error accessing WS cache for reconciliation: {e}")
                reconciliation['errors'].append(f'Error accessing WS cache: {e}')
                # Proceed without WS cache comparison if error occurs

            # --- Get Exchange State via REST ---
            if perform_rest_check:
                reconciliation['rest_check_performed'] = True
                try:
                    self.logger.info("Reconciling positions using REST API...")
                    actual_positions_response = self.client.get_position_info()
                    if not actual_positions_response or actual_positions_response.get('retCode') != 0:
                        err = actual_positions_response.get('retMsg', 'Unknown') if actual_positions_response else 'No response'
                        reconciliation['errors'].append(f'Failed to get positions via REST: {err}')
                        self.logger.error(f"Reconciliation failed: Could not fetch positions via REST ({err}).")
                        # Cannot proceed with comparison if REST fails
                        return {'success': False, 'reconciliation': reconciliation}
                    else:
                        rest_fetch_successful = True
                        # Parse REST response
                        for pos in actual_positions_response['result'].get('list', []):
                            symbol = pos.get('symbol')
                            size = float(pos.get('size', 0))
                            if symbol: # Store even zero-size positions from REST for comparison
                                actual_positions_rest[symbol] = {
                                    'size': size, 'side': pos.get('side'),
                                    'avgPrice': float(pos.get('avgPrice', 0)),
                                    'positionValue': float(pos.get('positionValue', 0)),
                                    'updatedTime': int(pos.get('updatedTime', time.time()*1000))
                                }
                        reconciliation['rest_count'] = len(actual_positions_rest)
                        self.logger.info(f"REST reconciliation: Found {len([p for p in actual_positions_rest.values() if p['size'] > 0])} open positions.")

                        # --- Update WS cache with REST data for synchronization ---
                        with self._state_lock:
                            # Overwrite cache with REST truth, including zero positions found by REST
                            self.position_cache = actual_positions_rest.copy()
                            # Remove symbols from cache that were NOT in the REST response at all
                            symbols_in_rest = set(actual_positions_rest.keys())
                            symbols_in_cache = list(self.position_cache.keys()) # Iterate over copy of keys
                            for sym in symbols_in_cache:
                                if sym not in symbols_in_rest:
                                    del self.position_cache[sym]
                            self.logger.info("Internal position cache synchronized with REST API data.")

                except Exception as e:
                    reconciliation['errors'].append(f'Exception during REST reconciliation: {e}')
                    self.logger.error(f"Exception during REST reconciliation: {e}", exc_info=True)
                    return {'success': False, 'reconciliation': reconciliation}
            else:
                # If not performing REST check, we can't truly reconcile. Just report cache state.
                self.logger.info("Skipping REST check in reconcile_positions. Reporting WS cache state only.")
                reconciliation['matched'] = list(ws_cache_snapshot.keys()) # Assume cache is correct if not checking REST
                return {'success': True, 'reconciliation': reconciliation}


            # --- Compare WS Snapshot vs REST Data ---
            if rest_fetch_successful:
                all_symbols = set(ws_cache_snapshot.keys()) | set(actual_positions_rest.keys())

                for symbol in all_symbols:
                    ws_pos = ws_cache_snapshot.get(symbol)
                    rest_pos = actual_positions_rest.get(symbol)

                    # Clean up REST position if size is effectively zero
                    if rest_pos and abs(rest_pos['size']) < 1e-9:
                        rest_pos = None # Treat negligible size as closed

                    # Clean up WS position if size is effectively zero
                    if ws_pos and abs(ws_pos['size']) < 1e-9:
                        ws_pos = None


                    if ws_pos and rest_pos:
                        # Compare size, side etc. Use tolerance for float comparison.
                        size_diff = abs(ws_pos['size'] - rest_pos['size'])
                        side_match = ws_pos['side'] == rest_pos['side']
                        # Price comparison is less critical for reconciliation, focus on size/side
                        # price_diff = abs(ws_pos['avgPrice'] - rest_pos['avgPrice'])

                        # Define mismatch threshold (e.g., 0.01% of size or a tiny absolute value)
                        mismatch_threshold = max(1e-6, rest_pos['size'] * 0.0001)

                        if size_diff > mismatch_threshold or not side_match:
                            reconciliation['mismatched_details'].append({
                                'symbol': symbol,
                                'ws_state': ws_pos,
                                'rest_state': rest_pos,
                                'size_diff': size_diff,
                                'side_mismatch': not side_match
                            })
                        else:
                            reconciliation['matched'].append(symbol)
                    elif ws_pos and not rest_pos: # In WS cache (size > 0), but not in REST (or size is 0)
                        reconciliation['missing_on_rest'].append({'symbol': symbol, 'ws_state': ws_pos})
                    elif not ws_pos and rest_pos: # Not in WS cache (or size 0), but found open in REST
                        reconciliation['unexpected_on_rest'].append({'symbol': symbol, 'rest_state': rest_pos})
                    # Else: both are None (size 0), which is consistent


            # --- Report Results ---
            mismatched_count = len(reconciliation['mismatched_details'])
            missing_count = len(reconciliation['missing_on_rest'])
            unexpected_count = len(reconciliation['unexpected_on_rest'])
            total_issues = mismatched_count + missing_count + unexpected_count

            if total_issues > 0:
                self.logger.warning(f"Position reconciliation result: "
                            f"{len(reconciliation['matched'])} matched, "
                            f"{mismatched_count} mismatched, "
                            f"{missing_count} missing on REST, "
                            f"{unexpected_count} unexpected on REST.")
                # Trigger further action if needed, e.g., alert, forced sync, emergency stop
                if mismatched_count > 1 or missing_count > 1 or unexpected_count > 1:
                    self.logger.error("Significant position discrepancies found during reconciliation! Manual review may be needed.")
                    # self.emergency_protocols.execute_emergency_stop(...) # Potentially trigger emergency
            else:
                self.logger.info("Position reconciliation successful: Internal cache matches exchange state.")

            return {'success': not reconciliation['errors'], 'reconciliation': reconciliation}

    def emergency_stop_with_verification(self) -> Dict:
            """Orchestrates emergency stop: cancel orders, close positions, verify."""
            # (Implementation provided previously is mostly correct, ensure logging and updated calls)
            self.logger.critical("🚨 EMERGENCY STOP WITH VERIFICATION ACTIVATED!")
            start_time = time.time()
            results = {'success': False, 'message': 'Emergency stop initiated'}
            try:
                # 1. Cancel All Orders
                cancel_result = self.cancel_all_orders() # Uses updated method
                if not cancel_result or cancel_result.get('retCode') != 0:
                    self.logger.error("Failed to cancel all orders during emergency stop.")
                    # Continue anyway

                # 2. Identify Open Positions (Use REST for safety)
                symbols_to_close = []
                try:
                    current_positions_response = self.client.get_position_info()
                    if current_positions_response and current_positions_response.get('retCode') == 0:
                        symbols_to_close = [p['symbol'] for p in current_positions_response['result'].get('list', []) if float(p.get('size', 0)) > 0]
                    else:
                        err = current_positions_response.get('retMsg','Unknown') if current_positions_response else 'No response'
                        self.logger.error(f"Could not get current positions for emergency stop via REST: {err}. Falling back to WS cache.")
                        with self._state_lock: symbols_to_close = [sym for sym, data in self.position_cache.items() if data.get('size', 0) > 0]
                except Exception as pos_e:
                    self.logger.error(f"Exception getting positions for emergency stop: {pos_e}. Using known symbols as fallback.")
                    symbols_to_close = list(SYMBOLS)

                if not symbols_to_close:
                    self.logger.info("No open positions found to close during emergency stop.")
                    results.update({'success': True, 'message': 'No open positions found.'})
                    return results

                self.logger.info(f"Attempting guaranteed closure for symbols: {symbols_to_close}")

                # 3. Execute Guaranteed Closure for each symbol
                closure_results = {}
                failed_closures = []
                for symbol in set(symbols_to_close):
                    self.logger.info(f"Emergency Closing {symbol}...")
                    result = self.execute_guaranteed_position_closure(symbol, max_attempts=3) # Uses updated method
                    closure_results[symbol] = result
                    if not result.get('success'):
                        failed_closures.append(symbol)
                        self.logger.error(f"❌ Guaranteed closure FAILED for {symbol}: {result.get('message')}")
                    # Verification is now part of guaranteed_closure
                    time.sleep(0.3)

                # 4. Final Verification Loop (Redundant if guaranteed_closure verifies, but safe)
                verification_results = {}
                all_verified_closed = True
                final_check_symbols = set(symbols_to_close)
                self.logger.info(f"Performing FINAL position verification for: {final_check_symbols}")
                for symbol in final_check_symbols:
                    is_closed = self._verify_position_closure(symbol, timeout=20) # Use verification
                    verification_results[symbol] = is_closed
                    if not is_closed:
                        self.logger.critical(f"❌ FINAL VERIFICATION FAILED for {symbol} - POSITION MAY REMAIN OPEN!")
                        all_verified_closed = False
                    else:
                        self.logger.info(f"✅ Final verification passed for {symbol}.")
                    time.sleep(0.1)

                # 5. Final Reconciliation
                final_reconciliation = self.reconcile_positions(perform_rest_check=True) # Force REST check

                duration = time.time() - start_time
                results = {
                    'success': all_verified_closed,
                    'message': 'Emergency stop completed. Check verification results.' if all_verified_closed else 'Emergency stop completed WITH FAILURES. MANUAL INTERVENTION REQUIRED.',
                    'duration_seconds': duration,
                    'closure_attempts': closure_results,
                    'final_verification': verification_results,
                    'final_reconciliation': final_reconciliation.get('reconciliation', {})
                }

                if all_verified_closed:
                    self.logger.info(f"✅ Emergency stop completed successfully in {duration:.2f}s.")
                else:
                    self.logger.critical(f"⚠️ Emergency stop completed in {duration:.2f}s WITH {len(failed_closures)} FAILED CLOSURES and/or verification failures. Manual intervention likely required.")

                return results

            except Exception as e:
                self.logger.critical(f"❌ Exception during emergency_stop_with_verification: {e}", exc_info=True)
                return {'success': False, 'error': str(e), 'message': 'Exception during emergency stop process.'}

    def close_position(self, symbol: str, side: str = None):
            """Places a market order to close the specified position."""
            # Side parameter is ignored, closure side is determined by current position
            self.logger.info(f"Attempting to close position for {symbol}...")
            try:
                # Get current position details reliably via REST before closing
                position_response = self.client.get_position_info(symbol)
                if not position_response or position_response.get('retCode') != 0:
                    # Try checking WS cache as a fallback ONLY if REST fails critically
                    with self._state_lock:
                        cached_pos = self.position_cache.get(symbol)
                    if cached_pos and cached_pos['size'] != 0:
                        self.logger.warning("REST failed for get_position_info, using WS cache data for closure (RISKY).")
                        position = cached_pos # Use cache data cautiously
                    else:
                        self.logger.error(f"Failed to get position info for {symbol} to close: {position_response.get('retMsg', 'Unknown') if position_response else 'No response'}")
                        return {'success': False, 'message': 'Failed to get position info'}
                else:
                    positions = position_response['result'].get('list', [])
                    position = next((p for p in positions if p['symbol'] == symbol), None)


                if not position or float(position.get('size', 0)) == 0:
                    self.logger.info(f"No open position found for {symbol} to close.")
                    # Ensure cache reflects this
                    with self._state_lock:
                        if symbol in self.position_cache:
                            self.position_cache[symbol]['size'] = 0
                            self.position_cache[symbol]['updatedTime'] = time.time()*1000
                    return {'success': True, 'message': 'No position found'}

                # Determine close side and quantity from reliable source (REST preferred)
                # Ensure size is fetched correctly (might be string in REST response)
                current_side = position['side'] # 'Buy' or 'Sell'
                close_side = 'Buy' if current_side == 'Sell' else 'Sell'
                quantity = float(position['size']) # Size is always positive

                self.logger.info(f"Closing {symbol} position: {quantity} units via {close_side} market order.")

                order_response = self.client.place_order(
                    symbol=symbol,
                    side=close_side,
                    order_type='Market',
                    qty=quantity,
                    price=None # Not needed for market
                    # Add reduceOnly=True if supported and desired? Bybit v5 doesn't explicitly list it for Market.
                )

                if order_response and order_response.get('retCode') == 0:
                    order_id = order_response['result']['orderId']
                    self.logger.info(f"✅ Position close order placed successfully for {symbol}. Order ID: {order_id}")

                    # --- Update Internal State Optimistically ---
                    with self._state_lock:
                        # Remove from open orders if tracked (market should fill fast)
                        if order_id in self.open_orders: del self.open_orders[order_id]

                        # Update position cache size to 0
                        if symbol in self.position_cache:
                            self.position_cache[symbol]['size'] = 0
                            # Update timestamp roughly, WS will give precise update
                            self.position_cache[symbol]['updatedTime'] = time.time()*1000
                        else: # Add entry if somehow missing
                            self.position_cache[symbol] = {'size': 0, 'updatedTime': time.time()*1000}
                    # ------------------------------------------

                    return {
                        'success': True,
                        'order_id': order_id,
                        'symbol': symbol,
                        'side': close_side, # Side of the closing order
                        'quantity': quantity
                    }
                else:
                    error_msg = order_response.get('retMsg', 'Unknown error') if order_response else 'No response'
                    self.logger.error(f"Failed to place close order for {symbol}: {error_msg}")
                    return {'success': False, 'message': f'Failed to close position: {error_msg}'}

            except Exception as e:
                self.logger.error(f"Exception closing position for {symbol}: {e}", exc_info=True)
                return {'success': False, 'message': f'Error closing position: {e}'}
        
    def get_trade_history(self, limit: int = 50):
            """Returns a copy of the recent trade history stored locally."""
            with self._state_lock: # Protect access if modified elsewhere
                # Make sure trade_history structure matches what's expected
                history_copy = self.trade_history[:] # Shallow copy
            if limit >= len(history_copy):
                return history_copy
            return history_copy[-limit:]
    
    def get_performance_metrics(self):
            """Calculates basic performance metrics from local trade history."""
            # Note: This relies on local history and simple 'success' status.
            # For accurate PNL-based metrics, querying the database is better.
            local_history = self.get_trade_history(limit=500) # Use more history for metrics
            if not local_history:
                self.logger.warning("get_performance_metrics: No local trade history available.")
                return {}

            total_trades = len(local_history)
            # Count successful placements as 'wins' for this simple metric
            successful_placements = [t for t in local_history if t.get('success', False)]
            win_rate_placeholder = len(successful_placements) / total_trades * 100 if total_trades > 0 else 0

            # Calculate averages only from relevant trades
            confidences = [t.get('confidence', 0) for t in local_history if t.get('confidence') is not None]
            risk_rewards = [t.get('risk_reward_ratio', 0) for t in local_history if t.get('risk_reward_ratio') is not None]

            avg_confidence = np.mean(confidences) if confidences else 0
            avg_risk_reward = np.mean(risk_rewards) if risk_rewards else 0

            metrics = {
                'total_trades': total_trades,
                'win_rate': win_rate_placeholder, # Placeholder - needs PNL
                'avg_confidence': avg_confidence,
                'avg_risk_reward': avg_risk_reward,
                'recent_trades': local_history[-10:] # Show last 10 attempts
            }
            self.logger.debug(f"Calculated performance metrics from local history: {metrics}")
            return metrics
    
    def emergency_stop(self):
        return self.emergency_stop_with_verification()

class MarketImpactModel:
    def __init__(self):
        self.impact_cache = {}
        self.historical_impact = {}
        
    def estimate_impact(self, symbol: str, quantity: float, side: str) -> float:
        cache_key = f"{symbol}_{quantity}_{side}"
        if cache_key in self.impact_cache:
            return self.impact_cache[cache_key]
        
        try:
            orderbook = self._get_orderbook(symbol)
            if not orderbook:
                return self._fallback_impact(quantity, side)
            
            if side == 'Buy':
                levels = orderbook['result']['a']
                impact = self._calculate_slippage(levels, quantity, 'ask')
            else:
                levels = orderbook['result']['b']
                impact = self._calculate_slippage(levels, quantity, 'bid')
            
            self.impact_cache[cache_key] = impact
            return impact
            
        except Exception as e:
            return self._fallback_impact(quantity, side)

    def _get_orderbook(self, symbol: str):
        try:
            from bybit_client import BybitClient
            client = BybitClient()
            return client.get_orderbook(symbol)
        except:
            return None

    def _calculate_slippage(self, levels: List, quantity: float, side: str) -> float:
        remaining_qty = quantity
        total_cost = 0
        base_price = float(levels[0][0])
        
        for level in levels:
            price = float(level[0])
            size = float(level[1])
            
            if remaining_qty <= 0:
                break
            
            if remaining_qty <= size:
                total_cost += remaining_qty * price
                remaining_qty = 0
            else:
                total_cost += size * price
                remaining_qty -= size
        
        if quantity == 0:
            return 0
        
        avg_price = total_cost / quantity
        slippage = (avg_price - base_price) / base_price * 100
        
        if side == 'bid':
            slippage = -slippage
        
        return abs(slippage)

    def _fallback_impact(self, quantity: float, side: str) -> float:
        base_impact = 0.01
        quantity_impact = min(quantity * 0.001, 0.1)
        
        if side == 'Buy':
            impact = base_impact + quantity_impact
        else:
            impact = base_impact + quantity_impact * 0.8
        
        return impact

class LimitOrderStrategies:
    def __init__(self):
        self.strategies = {
            'aggressive': 0.0005,
            'neutral': 0.001,
            'conservative': 0.002,
            'iceberg': 0.0002
        }
    
    def calculate_limit_price(self, symbol: str, side: str, current_price: float, 
                            strategy: str, quantity: float) -> float:
        spread = self.strategies.get(strategy, 0.001)
        
        if side == 'Buy':
            return current_price * (1 - spread)
        else:
            return current_price * (1 + spread)

class VWAPExecutor:
    def __init__(self, client: BybitClient):
        self.client = client
        self.active_executions = {}
        
    def execute_vwap(self, symbol: str, action: str, quantity: float, 
                    decision: Dict, duration: int = 300) -> Dict:
        try:
            print(f"📊 Executing VWAP for {symbol}: {quantity} units over {duration}s")
            
            side = 'Buy' if action == 'BUY' else 'Sell'
            chunks = self._calculate_vwap_chunks(quantity, duration)
            total_executed = 0
            executed_orders = []
            
            start_time = time.time()
            chunk_times = np.linspace(0, duration, len(chunks))
            
            for i, (chunk_qty, chunk_time) in enumerate(zip(chunks, chunk_times)):
                elapsed = time.time() - start_time
                if elapsed < chunk_time:
                    time.sleep(chunk_time - elapsed)
                
                current_price = self._get_current_price(symbol)
                limit_price = self._calculate_vwap_price(symbol, side, current_price)
                
                order_response = self.client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type='Limit',
                    qty=chunk_qty,
                    price=limit_price,
                    time_in_force='ImmediateOrCancel'
                )
                
                if order_response and order_response.get('retCode') == 0:
                    executed_orders.append({
                        'order_id': order_response['result']['orderId'],
                        'quantity': chunk_qty,
                        'price': limit_price,
                        'timestamp': time.time()
                    })
                    total_executed += chunk_qty
                    print(f"✅ VWAP chunk {i+1} executed: {chunk_qty} units")
                else:
                    print(f"⚠️ VWAP chunk {i+1} failed")
            
            if total_executed < quantity * 0.95:
                remaining = quantity - total_executed
                market_response = self.client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type='Market',
                    qty=remaining,
                    price=None
                )
                
                if market_response and market_response.get('retCode') == 0:
                    total_executed += remaining
                    print(f"✅ VWAP completed with market order for remaining {remaining} units")
            
            return {
                'success': True,
                'strategy': 'VWAP',
                'total_executed': total_executed,
                'completion_rate': total_executed / quantity,
                'executed_orders': executed_orders
            }
            
        except Exception as e:
            return {'success': False, 'message': f'VWAP execution error: {e}'}

    def _calculate_vwap_chunks(self, total_quantity: float, duration: int) -> List[float]:
        num_chunks = max(3, min(10, int(duration / 30)))
        chunks = []
        
        for i in range(num_chunks):
            if i < num_chunks - 1:
                chunk = total_quantity * (0.6 if i == 0 else 0.8 / (num_chunks - 1))
            else:
                chunk = total_quantity - sum(chunks)
            chunks.append(max(0.001, chunk))
        
        return chunks

    def _calculate_vwap_price(self, symbol: str, side: str, current_price: float) -> float:
        orderbook = self.client.get_orderbook(symbol)
        if not orderbook or not orderbook.get('result'):
            return current_price
        
        if side == 'Buy':
            levels = orderbook['result']['a'][:3]
            weights = [0.5, 0.3, 0.2]
        else:
            levels = orderbook['result']['b'][:3]
            weights = [0.5, 0.3, 0.2]
        
        weighted_price = sum(float(levels[i][0]) * weights[i] for i in range(min(len(levels), len(weights))))
        return weighted_price

    def _get_current_price(self, symbol: str) -> float:
        try:
            ticker = self.client.get_ticker(symbol)
            if ticker and ticker.get('result'):
                return float(ticker['result']['list'][0]['lastPrice'])
        except:
            pass
        return 0.0

class TWAPExecutor:
    def __init__(self, client: BybitClient):
        self.client = client
        self.active_executions = {}
        
    def execute_twap(self, symbol: str, action: str, quantity: float,
                    decision: Dict, duration: int = 300) -> Dict:
        try:
            print(f"⏱️ Executing TWAP for {symbol}: {quantity} units over {duration}s")
            
            side = 'Buy' if action == 'BUY' else 'Sell'
            num_intervals = max(2, min(20, int(duration / 15)))
            chunk_size = quantity / num_intervals
            interval = duration / num_intervals
            
            total_executed = 0
            executed_orders = []
            
            for i in range(num_intervals):
                time.sleep(interval)
                
                current_price = self._get_current_price(symbol)
                orderbook = self.client.get_orderbook(symbol)
                
                if orderbook and orderbook.get('result'):
                    if side == 'Buy':
                        limit_price = float(orderbook['result']['a'][0][0]) * 0.999
                    else:
                        limit_price = float(orderbook['result']['b'][0][0]) * 1.001
                else:
                    limit_price = current_price * (0.999 if side == 'Buy' else 1.001)
                
                order_response = self.client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type='Limit',
                    qty=chunk_size,
                    price=limit_price,
                    time_in_force='ImmediateOrCancel'
                )
                
                if order_response and order_response.get('retCode') == 0:
                    executed_orders.append({
                        'order_id': order_response['result']['orderId'],
                        'quantity': chunk_size,
                        'price': limit_price,
                        'timestamp': time.time()
                    })
                    total_executed += chunk_size
                    print(f"✅ TWAP interval {i+1} executed: {chunk_size} units")
                else:
                    print(f"⚠️ TWAP interval {i+1} failed")
            
            return {
                'success': True,
                'strategy': 'TWAP',
                'total_executed': total_executed,
                'completion_rate': total_executed / quantity,
                'executed_orders': executed_orders
            }
            
        except Exception as e:
            return {'success': False, 'message': f'TWAP execution error: {e}'}

    def _get_current_price(self, symbol: str) -> float:
        try:
            ticker = self.client.get_ticker(symbol)
            if ticker and ticker.get('result'):
                return float(ticker['result']['list'][0]['lastPrice'])
        except:
            pass
        return 0.0

class SmartOrderRouter:
    def __init__(self, client: BybitClient):
        self.client = client
        self.routing_strategies = ['limit', 'market', 'vwap', 'twap']
        
    def find_optimal_execution(self, symbol: str, action: str, quantity: float, 
                              orderbook_analysis: Dict) -> Dict:
        try:
            if not orderbook_analysis:
                return self._fallback_strategy(symbol, action, quantity)
            
            quality_score = orderbook_analysis.get('quality_score', 0.5)
            liquidity_density = orderbook_analysis.get('liquidity_density', 0.5)
            spread = orderbook_analysis.get('spread_bps', 50)
            
            if quality_score > 0.8 and liquidity_density > 0.7:
                strategy = 'limit'
                price = self._calculate_optimal_limit_price(symbol, action, orderbook_analysis)
            elif quality_score < 0.3 or spread > 100:
                strategy = 'market'
                price = orderbook_analysis.get('mid_price', 0)
            elif quantity > orderbook_analysis.get('total_bid_volume', 0) * 0.1:
                strategy = 'vwap' if quantity > 10000 else 'twap'
                price = orderbook_analysis.get('mid_price', 0)
            else:
                strategy = 'limit'
                price = self._calculate_optimal_limit_price(symbol, action, orderbook_analysis)
            
            estimated_impact = self._estimate_execution_impact(strategy, quantity, orderbook_analysis)
            
            return {
                'strategy': strategy,
                'price': price,
                'estimated_impact': estimated_impact,
                'quality': self._rate_execution_quality(estimated_impact),
                'orderbook_quality': quality_score
            }
            
        except Exception as e:
            return self._fallback_strategy(symbol, action, quantity)

    def _calculate_optimal_limit_price(self, symbol: str, action: str, orderbook: Dict) -> float:
        if action == 'BUY':
            weighted_ask = orderbook.get('weighted_ask', 0)
            return weighted_ask * 0.999
        else:
            weighted_bid = orderbook.get('weighted_bid', 0)
            return weighted_bid * 1.001

    def _estimate_execution_impact(self, strategy: str, quantity: float, orderbook: Dict) -> float:
        base_impacts = {
            'market': 0.05,
            'limit': 0.02,
            'vwap': 0.015,
            'twap': 0.018
        }
        
        base_impact = base_impacts.get(strategy, 0.03)
        liquidity_factor = 1 - orderbook.get('liquidity_density', 0.5)
        quantity_factor = min(quantity * 0.0001, 0.1)
        
        return base_impact * liquidity_factor + quantity_factor

    def _rate_execution_quality(self, impact: float) -> str:
        if impact < 0.02:
            return 'EXCELLENT'
        elif impact < 0.05:
            return 'GOOD'
        elif impact < 0.1:
            return 'AVERAGE'
        else:
            return 'POOR'

    def _fallback_strategy(self, symbol: str, action: str, quantity: float) -> Dict:
        return {
            'strategy': 'limit',
            'price': 0.0,
            'estimated_impact': 0.05,
            'quality': 'AVERAGE',
            'orderbook_quality': 0.5
        }