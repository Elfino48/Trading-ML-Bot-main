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
        self.open_orders = {}
        self.trade_history = []
        self.emergency_protocols = None
        self.position_timeout = 30
        self.max_retry_attempts = 5
        self.execution_quality_log = []
        self.market_impact_model = MarketImpactModel()
        self.limit_order_strategies = LimitOrderStrategies()
        self.vwap_executor = VWAPExecutor(bybit_client)
        self.twap_executor = TWAPExecutor(bybit_client)
        self.smart_router = SmartOrderRouter(bybit_client)
        
    def set_emergency_protocols(self, emergency_protocols):
        self.emergency_protocols = emergency_protocols
        
    def _validate_trade_decision(self, decision: Dict) -> bool:
        required_fields = ['symbol', 'action', 'quantity', 'position_size', 'current_price']
        
        for field in required_fields:
            if field not in decision:
                return False
        
        if decision['symbol'] not in SYMBOLS:
            return False
        
        if decision['action'] not in ['BUY', 'SELL', 'HOLD']:
            return False
        
        if decision['quantity'] <= 0:
            return False
        
        if decision['position_size'] <= 0:
            return False
        
        if decision['current_price'] <= 0:
            return False
        
        if decision['action'] != 'HOLD':
            if 'stop_loss' not in decision or 'take_profit' not in decision:
                return False
            
            if decision['stop_loss'] <= 0 or decision['take_profit'] <= 0:
                return False
            
            if decision['action'] == 'BUY':
                if decision['stop_loss'] >= decision['current_price']:
                    return False
                if decision['take_profit'] <= decision['current_price']:
                    return False
            else:
                if decision['stop_loss'] <= decision['current_price']:
                    return False
                if decision['take_profit'] >= decision['current_price']:
                    return False
        
        return True

    def execute_limit_order(self, decision: Dict, strategy: str = 'aggressive') -> Dict:
        try:
            if not self._validate_trade_decision(decision):
                return {'success': False, 'message': 'Trade validation failed'}
            
            symbol = decision['symbol']
            action = decision['action']
            
            if action == 'HOLD':
                return {'success': True, 'message': 'No action needed'}
            
            if self.emergency_protocols and self.emergency_protocols.emergency_mode:
                return {'success': False, 'message': 'Emergency mode active'}
            
            current_price = decision['current_price']
            quantity = decision['quantity']
            position_size_usdt = decision['position_size']
            
            risk_approval = self.risk_manager.can_trade(symbol, position_size_usdt)
            
            if not risk_approval['approved']:
                return {'success': False, 'message': f'Risk management rejected trade: {risk_approval["reason"]}'}
            
            orderbook_analysis = self._analyze_order_book(symbol)
            optimal_execution = self.smart_router.find_optimal_execution(symbol, action, quantity, orderbook_analysis)
            
            side = 'Buy' if action == 'BUY' else 'Sell'
            
            if optimal_execution['strategy'] == 'limit':
                limit_price = optimal_execution['price']
                market_impact = optimal_execution['estimated_impact']
                
                print(f"🎯 Executing LIMIT {action} for {symbol} at ${limit_price:.2f}")
                print(f"   Strategy: {strategy}, Impact: {market_impact:.4f}%")
                
                order_response = self.client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type='Limit',
                    qty=quantity,
                    price=limit_price,
                    stop_loss=decision['stop_loss'],
                    take_profit=decision['take_profit'],
                    time_in_force='GoodTillCancel'
                )
                
                if order_response and order_response.get('retCode') == 0:
                    order_id = order_response['result']['orderId']
                    
                    trade_record = {
                        'timestamp': time.time(),
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'limit_price': limit_price,
                        'size_usdt': position_size_usdt,
                        'order_id': order_id,
                        'stop_loss': decision['stop_loss'],
                        'take_profit': decision['take_profit'],
                        'order_type': 'LIMIT',
                        'strategy': strategy,
                        'market_impact': market_impact,
                        'confidence': decision['confidence'],
                        'composite_score': decision['composite_score'],
                        'risk_reward_ratio': decision['risk_reward_ratio'],
                        'execution_quality': optimal_execution['quality']
                    }
                    self.trade_history.append(trade_record)
                    
                    self.open_orders[symbol] = {
                        'order_id': order_id,
                        'side': side,
                        'quantity': quantity,
                        'limit_price': limit_price,
                        'strategy': strategy,
                        'timestamp': time.time()
                    }
                    
                    self._log_execution_quality('LIMIT_ORDER', symbol, current_price, limit_price, quantity, market_impact)
                    
                    return {
                        'success': True, 
                        'order_id': order_id,
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'limit_price': limit_price,
                        'strategy': strategy,
                        'execution_quality': optimal_execution['quality']
                    }
                else:
                    error_msg = order_response.get('retMsg', 'Unknown error') if order_response else 'No response'
                    return {'success': False, 'message': f'Limit order failed: {error_msg}'}
            else:
                return self._execute_algorithmic_order(decision, optimal_execution)
                
        except Exception as e:
            return {'success': False, 'message': f'Limit order error: {e}'}

    def execute_adaptive_trade(self, decision: Dict) -> Dict:
        try:
            symbol = decision['symbol']
            quantity = decision['quantity']
            current_price = decision['current_price']
            action = decision['action']
            
            if action == 'HOLD':
                return {'success': True, 'message': 'No action needed'}
            
            execution_quality = self._get_recent_execution_quality(symbol)
            market_conditions = self._analyze_market_conditions(symbol, quantity)
            orderbook_analysis = self._analyze_order_book(symbol)
            
            if market_conditions['high_volatility'] or execution_quality.get('poor_fill_rate', False):
                strategy = 'market'
            elif quantity > market_conditions['optimal_limit_size']:
                if quantity > 10000 / current_price:
                    strategy = 'vwap'
                else:
                    strategy = 'twap'
            else:
                strategy = 'smart_limit'
            
            if strategy == 'market':
                return self.execute_enhanced_trade(decision)
            elif strategy in ['vwap', 'twap']:
                return self._execute_algorithmic_order(decision, {'strategy': strategy})
            else:
                return self.execute_limit_order(decision, strategy)
                
        except Exception as e:
            return {'success': False, 'message': f'Adaptive trade error: {e}'}

    def _execute_algorithmic_order(self, decision: Dict, execution_plan: Dict) -> Dict:
        symbol = decision['symbol']
        action = decision['action']
        quantity = decision['quantity']
        current_price = decision['current_price']
        
        if execution_plan['strategy'] == 'vwap':
            return self.vwap_executor.execute_vwap(
                symbol, action, quantity, decision, execution_plan.get('duration', 300)
            )
        elif execution_plan['strategy'] == 'twap':
            return self.twap_executor.execute_twap(
                symbol, action, quantity, decision, execution_plan.get('duration', 300)
            )
        else:
            return self.execute_enhanced_trade(decision)

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
        try:
            if not self._validate_trade_decision(decision):
                return {'success': False, 'message': 'Trade validation failed'}
            
            symbol = decision['symbol']
            action = decision['action']
            
            if action == 'HOLD':
                return {'success': True, 'message': 'No action needed'}
            
            if self.emergency_protocols and self.emergency_protocols.emergency_mode:
                return {'success': False, 'message': 'Emergency mode active - trading suspended'}
            
            position_size_usdt = decision['position_size']
            quantity = decision['quantity']
            current_price = decision['current_price']
            
            risk_approval = self.risk_manager.can_trade(symbol, position_size_usdt)
            
            if not risk_approval['approved']:
                return {'success': False, 'message': f'Risk management rejected trade: {risk_approval["reason"]}'}
            
            print(f"🎯 Executing {action} for {symbol}")
            print(f"   📊 Quantity: {quantity:.4f} units (${position_size_usdt:.2f})")
            print(f"   🛡️ Stop Loss: ${decision['stop_loss']:.2f}")
            print(f"   🎯 Take Profit: ${decision['take_profit']:.2f}")
            print(f"   ⚖️ Risk/Reward: {decision['risk_reward_ratio']:.2f}:1")
            print(f"   📈 Confidence: {decision['confidence']:.1f}%")
            print(f"   🌡️ Market Regime: {decision['market_regime']}")
            
            if action == 'BUY':
                side = 'Buy'
                stop_loss = decision['stop_loss']
                take_profit = decision['take_profit']
            else:
                side = 'Sell' 
                stop_loss = decision['stop_loss']
                take_profit = decision['take_profit']
            
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
                    'take_profit': take_profit,
                    'confidence': decision['confidence'],
                    'composite_score': decision['composite_score'],
                    'risk_reward_ratio': decision['risk_reward_ratio'],
                    'market_regime': decision['market_regime'],
                    'volatility_regime': decision['volatility_regime'],
                    'ml_confidence': decision.get('ml_prediction', {}).get('confidence', 0)
                }
                self.trade_history.append(trade_record)
                
                self.open_orders[symbol] = {
                    'order_id': order_id,
                    'side': side,
                    'quantity': quantity,
                    'timestamp': time.time()
                }

                print(f"✅ Enhanced trade executed successfully!")
                
                return {
                    'success': True, 
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'position_size': position_size_usdt,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward': decision['risk_reward_ratio'],
                    'confidence': decision['confidence']
                }
            else:
                error_msg = order_response.get('retMsg', 'Unknown error') if order_response else 'No response'
                return {'success': False, 'message': f'Order failed: {error_msg}'}
                
        except Exception as e:
            return {'success': False, 'message': f'Execution error: {e}'}
    
    def execute_trade_with_retry(self, decision: Dict, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                if 'composite_score' in decision:
                    result = self.execute_enhanced_trade(decision)
                else:
                    result = self.execute_trade(decision)
                
                if result['success']:
                    return result
                else:
                    print(f"⚠️ Trade attempt {attempt + 1} failed: {result['message']}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        print(f"🔄 Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
            except Exception as e:
                print(f"❌ Exception in trade attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
        
        return {'success': False, 'message': f'All {max_retries} trade attempts failed'}
    
    def cancel_all_orders(self, symbol: str = None):
        try:
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol
                
            response = self.client._request("POST", "/v5/order/cancel-all", params)
            if response and response.get('retCode') == 0:
                print(f"✅ Cancelled all orders for {symbol if symbol else 'all symbols'}")
                if symbol and symbol in self.open_orders:
                    del self.open_orders[symbol]
                elif not symbol:
                    self.open_orders.clear()
            return response
        except Exception as e:
            print(f"❌ Error canceling orders: {e}")
            return None
    
    def get_position_info(self, symbol: str = None):
        try:
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol
                
            response = self.client._request("GET", "/v5/position/list", params)
            return response
        except Exception as e:
            print(f"❌ Error getting position info: {e}")
            return None
    
    def get_order_status(self, symbol: str, order_id: str):
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "orderId": order_id
            }
            
            response = self.client._request("GET", "/v5/order/history", params)
            return response
        except Exception as e:
            print(f"❌ Error getting order status: {e}")
            return None

    def execute_guaranteed_position_closure(self, symbol: str, max_attempts: int = 3) -> Dict:
        print(f"🛡️ Attempting guaranteed position closure for {symbol}")
        
        for attempt in range(max_attempts):
            try:
                result = self.close_position(symbol)
                if result.get('success'):
                    print(f"✅ Position closed successfully on attempt {attempt + 1}")
                    return result
                
                if attempt == 1:
                    result = self._force_close_position(symbol)
                    if result.get('success'):
                        print(f"✅ Position force-closed on attempt {attempt + 1}")
                        return result
                
                if attempt == 2:
                    result = self._emergency_position_closure(symbol)
                    if result.get('success'):
                        print(f"✅ Emergency position closure successful on attempt {attempt + 1}")
                        return result
                
                print(f"⚠️ Position closure attempt {attempt + 1} failed, retrying...")
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error in position closure attempt {attempt + 1}: {e}")
                time.sleep(2)
        
        return {'success': False, 'message': f'All {max_attempts} position closure attempts failed'}

    def _force_close_position(self, symbol: str) -> Dict:
        try:
            position_response = self.client.get_position_info(symbol)
            if not position_response or position_response.get('retCode') != 0:
                return {'success': False, 'message': 'Failed to get position info for force close'}
            
            positions = position_response['result']['list']
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position or float(position['size']) == 0:
                return {'success': True, 'message': 'No position found to force close'}
            
            current_side = position['side']
            close_side = 'Buy' if current_side == 'Sell' else 'Sell'
            quantity = float(position['size'])
            
            print(f"🔄 Force closing {symbol} position: {quantity} units via {close_side}")
            
            max_retries = 3
            for i in range(max_retries):
                try:
                    order_response = self.client.place_order(
                        symbol=symbol,
                        side=close_side,
                        order_type='Market',
                        qty=quantity,
                        price=None,
                        stop_loss=None,
                        take_profit=None
                    )
                    
                    if order_response and order_response.get('retCode') == 0:
                        order_id = order_response['result']['orderId']
                        print(f"✅ Force close successful. Order ID: {order_id}")
                        
                        if self._verify_position_closure(symbol):
                            if symbol in self.open_orders:
                                del self.open_orders[symbol]
                            return {
                                'success': True,
                                'order_id': order_id,
                                'method': 'force_close'
                            }
                        
                except Exception as e:
                    print(f"❌ Force close attempt {i + 1} failed: {e}")
                    if i < max_retries - 1:
                        quantity *= 0.8
                        print(f"🔄 Retrying with reduced quantity: {quantity}")
                        time.sleep(1)
            
            return {'success': False, 'message': 'Force close failed after all retries'}
            
        except Exception as e:
            return {'success': False, 'message': f'Error in force close: {e}'}

    def _emergency_position_closure(self, symbol: str) -> Dict:
        try:
            print(f"🚨 EMERGENCY closure for {symbol}")
            
            methods = [
                self._try_reduce_and_close,
                self._try_multiple_small_orders,
                self._try_different_order_types
            ]
            
            for method in methods:
                result = method(symbol)
                if result.get('success'):
                    return result
                
                time.sleep(1)
            
            return {'success': False, 'message': 'All emergency closure methods failed'}
            
        except Exception as e:
            return {'success': False, 'message': f'Emergency closure error: {e}'}

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

    def _verify_position_closure(self, symbol: str, timeout: int = 30) -> bool:
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                position_response = self.client.get_position_info(symbol)
                if position_response and position_response.get('retCode') == 0:
                    positions = position_response['result']['list']
                    position = next((p for p in positions if p['symbol'] == symbol), None)
                    
                    if not position or float(position.get('size', 0)) == 0:
                        print(f"✅ Verified position closure for {symbol}")
                        return True
                
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error verifying position closure: {e}")
                time.sleep(2)
        
        print(f"❌ Position closure verification timeout for {symbol}")
        return False

    def validate_order_execution(self, symbol: str, order_id: str, expected_side: str, expected_quantity: float) -> Dict:
        try:
            order_status = self.get_order_status(symbol, order_id)
            if not order_status or order_status.get('retCode') != 0:
                return {'valid': False, 'reason': 'Could not fetch order status'}
            
            order_info = order_status['result']['list'][0] if order_status['result']['list'] else None
            if not order_info:
                return {'valid': False, 'reason': 'Order not found'}
            
            actual_side = order_info.get('side')
            actual_quantity = float(order_info.get('qty', 0))
            order_status = order_info.get('orderStatus')
            
            validation_result = {
                'valid': True,
                'order_status': order_status,
                'side_match': actual_side == expected_side,
                'quantity_match': abs(actual_quantity - expected_quantity) < 0.001,
                'executed': order_status in ['Filled', 'PartiallyFilled']
            }
            
            validation_result['valid'] = (
                validation_result['side_match'] and 
                validation_result['quantity_match'] and
                validation_result['executed']
            )
            
            if not validation_result['valid']:
                validation_result['reason'] = f"Validation failed: side_match={validation_result['side_match']}, quantity_match={validation_result['quantity_match']}, executed={validation_result['executed']}"
            
            return validation_result
            
        except Exception as e:
            return {'valid': False, 'reason': f'Validation error: {e}'}

    def reconcile_positions(self) -> Dict:
        try:
            reconciliation = {
                'matched': [],
                'mismatched': [],
                'unexpected': [],
                'missing': []
            }
            
            actual_positions_response = self.get_position_info()
            if not actual_positions_response or actual_positions_response.get('retCode') != 0:
                return {'success': False, 'error': 'Failed to get actual positions'}
            
            actual_positions = {}
            for pos in actual_positions_response['result']['list']:
                if float(pos.get('size', 0)) > 0:
                    actual_positions[pos['symbol']] = {
                        'size': float(pos['size']),
                        'side': pos['side'],
                        'entry_price': float(pos.get('avgPrice', 0))
                    }
            
            for symbol, expected_data in self.open_orders.items():
                actual_data = actual_positions.get(symbol)
                
                if actual_data:
                    if symbol in actual_positions:
                        reconciliation['matched'].append(symbol)
                    else:
                         reconciliation['mismatched'].append({
                            'symbol': symbol,
                            'expected': expected_data,
                            'actual': actual_data
                         })
                else:
                    reconciliation['missing'].append(symbol)
            
            for symbol in actual_positions:
                if symbol not in self.open_orders:
                    reconciliation['unexpected'].append({
                        'symbol': symbol,
                        'position': actual_positions[symbol]
                    })
            
            if reconciliation['mismatched'] or reconciliation['missing'] or reconciliation['unexpected']:
                print(f"🔄 Position reconciliation: {len(reconciliation['matched'])} matched, "
                      f"{len(reconciliation['mismatched'])} mismatched, "
                      f"{len(reconciliation['missing'])} missing, "
                      f"{len(reconciliation['unexpected'])} unexpected")
            
            return {'success': True, 'reconciliation': reconciliation}
            
        except Exception as e:
            return {'success': False, 'error': f'Reconciliation failed: {e}'}

    def emergency_stop_with_verification(self) -> Dict:
        print("🚨 EMERGENCY STOP WITH VERIFICATION ACTIVATED!")
        
        try:
            self.cancel_all_orders()
            print("✅ All orders cancelled")
            
            closure_results = {}
            current_positions_response = self.get_position_info()
            symbols_to_close = []
            if current_positions_response and current_positions_response.get('retCode') == 0:
                 symbols_to_close = [p['symbol'] for p in current_positions_response['result']['list'] if float(p.get('size', 0)) > 0]
            
            for symbol in set(SYMBOLS) | set(symbols_to_close):
                print(f"🔄 Closing position for {symbol}...")
                result = self.execute_guaranteed_position_closure(symbol)
                closure_results[symbol] = result
                
                if result.get('success'):
                    print(f"✅ Position closed for {symbol}")
                else:
                    print(f"❌ Failed to close position for {symbol}: {result.get('message')}")
                
                time.sleep(0.2)
            
            verification_results = {}
            symbols_to_verify = set(SYMBOLS) | set(closure_results.keys())
            all_closed = True
            for symbol in symbols_to_verify:
                is_closed = self._verify_position_closure(symbol, timeout=5)
                verification_results[symbol] = is_closed
                
                if not is_closed:
                    print(f"❌ Position verification failed for {symbol} - position remains open!")
                    all_closed = False
            
            final_reconciliation = self.reconcile_positions()
            
            emergency_result = {
                'success': all_closed,
                'closure_results': closure_results,
                'verification_results': verification_results,
                'reconciliation': final_reconciliation
            }
            
            if emergency_result['success']:
                print("✅ Emergency stop completed successfully - all positions verified closed")
            else:
                print("⚠️ Emergency stop completed with some verification failures. Manual intervention may be required.")
            
            return emergency_result
            
        except Exception as e:
            print(f"❌ Error during emergency stop: {e}")
            return {'success': False, 'error': str(e)}

    def close_position(self, symbol: str, side: str = None):
        try:
            position_response = self.get_position_info(symbol)
            if not position_response or position_response.get('retCode') != 0:
                return {'success': False, 'message': 'Failed to get position info'}
            
            positions = position_response['result']['list']
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position or float(position['size']) == 0:
                return {'success': True, 'message': 'No position found'}
            
            current_side = position['side']
            close_side = 'Buy' if current_side == 'Sell' else 'Sell'
            quantity = float(position['size'])
            
            print(f"Closing {symbol} position: {quantity} units via {close_side}")
            
            order_response = self.client.place_order(
                symbol=symbol,
                side=close_side,
                order_type='Market',
                qty=quantity,
                price=None
            )
            
            if order_response and order_response.get('retCode') == 0:
                order_id = order_response['result']['orderId']
                print(f"✅ Position closed successfully. Order ID: {order_id}")
                
                if symbol in self.open_orders:
                    del self.open_orders[symbol]
                
                return {
                    'success': True,
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': close_side,
                    'quantity': quantity
                }
            else:
                error_msg = order_response.get('retMsg', 'Unknown error') if order_response else 'No response'
                return {'success': False, 'message': f'Failed to close position: {error_msg}'}
                
        except Exception as e:
            return {'success': False, 'message': f'Error closing position: {e}'}
    
    def get_trade_history(self, limit: int = 50):
        if limit > len(self.trade_history):
            return self.trade_history
        return self.trade_history[-limit:]
    
    def get_performance_metrics(self):
        if not self.trade_history:
            return {}
        
        total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if 
                          (t['side'] == 'Buy' and t.get('take_profit', 0) > t['price']) or
                          (t['side'] == 'Sell' and t.get('take_profit', 0) < t['price'])]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        avg_confidence = sum(t.get('confidence', 0) for t in self.trade_history) / total_trades
        
        avg_risk_reward = sum(t.get('risk_reward_ratio', 0) for t in self.trade_history) / total_trades
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_confidence': avg_confidence,
            'avg_risk_reward': avg_risk_reward,
            'recent_trades': self.trade_history[-10:]
        }
    
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