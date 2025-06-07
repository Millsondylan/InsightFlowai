import hashlib
import jwt
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
from config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRATION

logger = logging.getLogger(__name__)

def generate_token(user_id: int) -> str:
    """Generate JWT token for user"""
    try:
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=JWT_EXPIRATION)
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    except Exception as e:
        logger.error(f"Error generating token: {e}")
        return None

def verify_token(token: str) -> Optional[int]:
    """Verify JWT token and return user_id"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get('user_id')
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {e}")
        return None

def generate_referral_code(length: int = 8) -> str:
    """Generate unique referral code"""
    try:
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choice(chars) for _ in range(length))
    except Exception as e:
        logger.error(f"Error generating referral code: {e}")
        return None

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    try:
        return hashlib.sha256(password.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Error hashing password: {e}")
        return None

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    try:
        return hash_password(password) == hashed
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False

def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format currency amount"""
    try:
        return f"{currency} {amount:,.2f}"
    except Exception as e:
        logger.error(f"Error formatting currency: {e}")
        return str(amount)

def format_percentage(value: float) -> str:
    """Format percentage value"""
    try:
        return f"{value:,.2f}%"
    except Exception as e:
        logger.error(f"Error formatting percentage: {e}")
        return str(value)

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp"""
    try:
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"Error formatting timestamp: {e}")
        return str(timestamp)

def calculate_pnl(entry_price: float, current_price: float, quantity: float) -> Dict[str, float]:
    """Calculate profit/loss"""
    try:
        pnl = (current_price - entry_price) * quantity
        pnl_percentage = (pnl / (entry_price * quantity)) * 100
        return {
            'pnl': pnl,
            'pnl_percentage': pnl_percentage
        }
    except Exception as e:
        logger.error(f"Error calculating PnL: {e}")
        return {'pnl': 0.0, 'pnl_percentage': 0.0}

def calculate_position_size(account_balance: float, risk_percentage: float, stop_loss: float, entry_price: float) -> float:
    """Calculate position size based on risk management"""
    try:
        risk_amount = account_balance * (risk_percentage / 100)
        risk_per_share = abs(entry_price - stop_loss)
        return risk_amount / risk_per_share
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return 0.0

def calculate_risk_reward_ratio(entry_price: float, stop_loss: float, take_profit: float) -> float:
    """Calculate risk/reward ratio"""
    try:
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        return reward / risk if risk != 0 else 0
    except Exception as e:
        logger.error(f"Error calculating risk/reward ratio: {e}")
        return 0.0

def calculate_moving_average(prices: List[float], period: int) -> List[float]:
    """Calculate moving average"""
    try:
        if len(prices) < period:
            return []
        return [sum(prices[i:i+period])/period for i in range(len(prices)-period+1)]
    except Exception as e:
        logger.error(f"Error calculating moving average: {e}")
        return []

def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Calculate Relative Strength Index"""
    try:
        if len(prices) < period + 1:
            return []
        
        deltas = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return []

def calculate_bollinger_bands(prices: List[float], period: int = 20, num_std: float = 2.0) -> Dict[str, List[float]]:
    """Calculate Bollinger Bands"""
    try:
        if len(prices) < period:
            return {'middle': [], 'upper': [], 'lower': []}
        
        ma = calculate_moving_average(prices, period)
        std = []
        
        for i in range(len(ma)):
            start_idx = i
            end_idx = i + period
            period_prices = prices[start_idx:end_idx]
            std.append((sum((x - ma[i])**2 for x in period_prices) / period)**0.5)
        
        upper = [ma[i] + (num_std * std[i]) for i in range(len(ma))]
        lower = [ma[i] - (num_std * std[i]) for i in range(len(ma))]
        
        return {
            'middle': ma,
            'upper': upper,
            'lower': lower
        }
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return {'middle': [], 'upper': [], 'lower': []}

def calculate_macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, List[float]]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    try:
        if len(prices) < slow_period + signal_period:
            return {'macd': [], 'signal': [], 'histogram': []}
        
        fast_ma = calculate_moving_average(prices, fast_period)
        slow_ma = calculate_moving_average(prices, slow_period)
        
        macd_line = [fast_ma[i] - slow_ma[i] for i in range(len(slow_ma))]
        signal_line = calculate_moving_average(macd_line, signal_period)
        
        histogram = [macd_line[i] - signal_line[i] for i in range(len(signal_line))]
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return {'macd': [], 'signal': [], 'histogram': []}

def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels"""
    try:
        diff = high - low
        return {
            '0.0': low,
            '0.236': low + 0.236 * diff,
            '0.382': low + 0.382 * diff,
            '0.5': low + 0.5 * diff,
            '0.618': low + 0.618 * diff,
            '0.786': low + 0.786 * diff,
            '1.0': high
        }
    except Exception as e:
        logger.error(f"Error calculating Fibonacci levels: {e}")
        return {}

def calculate_support_resistance(prices: List[float], window: int = 20) -> Dict[str, List[float]]:
    """Calculate support and resistance levels"""
    try:
        if len(prices) < window:
            return {'support': [], 'resistance': []}
        
        support_levels = []
        resistance_levels = []
        
        for i in range(window, len(prices) - window):
            window_prices = prices[i-window:i+window]
            if prices[i] == min(window_prices):
                support_levels.append(prices[i])
            if prices[i] == max(window_prices):
                resistance_levels.append(prices[i])
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    except Exception as e:
        logger.error(f"Error calculating support/resistance: {e}")
        return {'support': [], 'resistance': []}

def format_market_data(data: Dict) -> str:
    """Format market data for display"""
    try:
        return f"""
Symbol: {data['symbol']}
Price: {format_currency(data['price'])}
Change: {format_currency(data['change'])} ({format_percentage(data['change_percent'])})
Volume: {data['volume']:,}
Last Updated: {format_timestamp(data['timestamp'])}
"""
    except Exception as e:
        logger.error(f"Error formatting market data: {e}")
        return str(data)

def format_portfolio_summary(portfolio: List[Dict]) -> str:
    """Format portfolio summary for display"""
    try:
        total_value = sum(item['current_price'] * item['quantity'] for item in portfolio)
        total_pnl = sum(calculate_pnl(item['entry_price'], item['current_price'], item['quantity'])['pnl'] for item in portfolio)
        
        return f"""
Portfolio Summary:
Total Value: {format_currency(total_value)}
Total P/L: {format_currency(total_pnl)}
Number of Positions: {len(portfolio)}

Positions:
{chr(10).join(f"{item['symbol']}: {item['quantity']} shares @ {format_currency(item['current_price'])}" for item in portfolio)}
"""
    except Exception as e:
        logger.error(f"Error formatting portfolio summary: {e}")
        return str(portfolio)

def format_alert(alert: Dict) -> str:
    """Format alert message"""
    try:
        return f"🔔 Alert: {alert['symbol']} {alert['condition']} {alert['price']}"
    except Exception as e:
        logger.error(f"Error formatting alert: {e}")
        return str(alert)

def format_error(error: str) -> str:
    """Format error message"""
    try:
        return f"❌ Error: {error}"
    except Exception as e:
        logger.error(f"Error formatting error message: {e}")
        return str(error)

def format_success_message(message: str) -> str:
    """Format success message"""
    try:
        return f"✅ Success: {message}"
    except Exception as e:
        logger.error(f"Error formatting success message: {e}")
        return str(message)

def format_warning_message(message: str) -> str:
    """Format warning message"""
    try:
        return f"⚠️ Warning: {message}"
    except Exception as e:
        logger.error(f"Error formatting warning message: {e}")
        return str(message)

def format_info_message(message: str) -> str:
    """Format info message"""
    try:
        return f"ℹ️ Info: {message}"
    except Exception as e:
        logger.error(f"Error formatting info message: {e}")
        return str(message)

# Aliases for compatibility
format_success = format_success_message
format_warning = format_warning_message
format_info = format_info_message

calculate_profit_loss = calculate_pnl
calculate_risk_reward = calculate_risk_reward_ratio 