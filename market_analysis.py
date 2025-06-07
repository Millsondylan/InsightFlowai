from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import asyncio
from api_integrations import MarketDataAPI
from database import Database
from utils import (
    calculate_moving_average, calculate_rsi, calculate_bollinger_bands,
    calculate_macd, calculate_fibonacci_levels, calculate_support_resistance,
    format_currency, format_percentage, format_timestamp
)

logger = logging.getLogger(__name__)

class MarketAnalysis:
    """Handles market data analysis"""
    
    def __init__(self, db: Database, market_api: MarketDataAPI):
        self.db = db
        self.market_api = market_api

    async def get_market_overview(self, user_id: int) -> Dict:
        """Get comprehensive market overview"""
        try:
            return await self.market_api.get_market_overview(user_id)
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {}

    async def get_stock_analysis(self, symbol: str, user_id: int) -> Dict:
        """Get detailed stock analysis"""
        try:
            # Get stock data
            stock_data = await self.market_api.get_stock_data(symbol, user_id)
            if not stock_data:
                return {}

            # Get technical indicators
            indicators = await self._get_technical_indicators(symbol, user_id)
            
            # Get support/resistance levels
            levels = await self._get_support_resistance(symbol, user_id)
            
            # Get market news
            news = await self.market_api.get_market_news(user_id)
            
            return {
                'symbol': symbol,
                'price_data': stock_data,
                'technical_indicators': indicators,
                'support_resistance': levels,
                'news': news,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting stock analysis: {e}")
            return {}

    async def get_crypto_analysis(self, symbol: str, user_id: int) -> Dict:
        """Get detailed cryptocurrency analysis"""
        try:
            # Get crypto data
            crypto_data = await self.market_api.get_crypto_data(symbol, user_id)
            if not crypto_data:
                return {}

            # Get technical indicators
            indicators = await self._get_technical_indicators(symbol, user_id)
            
            # Get support/resistance levels
            levels = await self._get_support_resistance(symbol, user_id)
            
            # Get market news
            news = await self.market_api.get_market_news(user_id)
            
            return {
                'symbol': symbol,
                'price_data': crypto_data,
                'technical_indicators': indicators,
                'support_resistance': levels,
                'news': news,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting crypto analysis: {e}")
            return {}

    async def get_forex_analysis(self, pair: str, user_id: int) -> Dict:
        """Get detailed forex analysis"""
        try:
            # Get forex data
            forex_data = await self.market_api.get_forex_data(pair, user_id)
            if not forex_data:
                return {}

            # Get technical indicators
            indicators = await self._get_technical_indicators(pair, user_id)
            
            # Get support/resistance levels
            levels = await self._get_support_resistance(pair, user_id)
            
            # Get market news
            news = await self.market_api.get_market_news(user_id)
            
            return {
                'pair': pair,
                'price_data': forex_data,
                'technical_indicators': indicators,
                'support_resistance': levels,
                'news': news,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting forex analysis: {e}")
            return {}

    async def _get_technical_indicators(self, symbol: str, user_id: int) -> Dict:
        """Get technical indicators for symbol"""
        try:
            # Get historical prices
            prices = await self._get_historical_prices(symbol, user_id)
            if not prices:
                return {}

            # Calculate indicators
            ma_20 = calculate_moving_average(prices, 20)
            ma_50 = calculate_moving_average(prices, 50)
            ma_200 = calculate_moving_average(prices, 200)
            
            rsi = calculate_rsi(prices)
            bb = calculate_bollinger_bands(prices)
            macd = calculate_macd(prices)
            
            return {
                'moving_averages': {
                    'ma_20': ma_20[-1] if ma_20 else None,
                    'ma_50': ma_50[-1] if ma_50 else None,
                    'ma_200': ma_200[-1] if ma_200 else None
                },
                'rsi': rsi[-1] if rsi else None,
                'bollinger_bands': {
                    'upper': bb['upper'][-1] if bb['upper'] else None,
                    'middle': bb['middle'][-1] if bb['middle'] else None,
                    'lower': bb['lower'][-1] if bb['lower'] else None
                },
                'macd': {
                    'macd': macd['macd'][-1] if macd['macd'] else None,
                    'signal': macd['signal'][-1] if macd['signal'] else None,
                    'histogram': macd['histogram'][-1] if macd['histogram'] else None
                }
            }
        except Exception as e:
            logger.error(f"Error getting technical indicators: {e}")
            return {}

    async def _get_support_resistance(self, symbol: str, user_id: int) -> Dict:
        """Get support and resistance levels"""
        try:
            # Get historical prices
            prices = await self._get_historical_prices(symbol, user_id)
            if not prices:
                return {}

            # Calculate levels
            levels = calculate_support_resistance(prices)
            
            # Calculate Fibonacci levels
            high = max(prices)
            low = min(prices)
            fib_levels = calculate_fibonacci_levels(high, low)
            
            return {
                'support': levels['support'],
                'resistance': levels['resistance'],
                'fibonacci': fib_levels
            }
        except Exception as e:
            logger.error(f"Error getting support/resistance: {e}")
            return {}

    async def _get_historical_prices(self, symbol: str, user_id: int) -> List[float]:
        """Get historical prices for symbol"""
        try:
            # Get historical data from database
            market_data = await self.db.get_market_data(symbol)
            if not market_data:
                return []

            # Extract prices
            return [float(price) for price in market_data['historical_prices']]
        except Exception as e:
            logger.error(f"Error getting historical prices: {e}")
            return []

    async def analyze_market_trend(self, symbol: str, timeframe: str, user_id: int) -> Dict:
        """Analyze market trend"""
        try:
            # Get technical indicators
            indicators = await self._get_technical_indicators(symbol, user_id)
            if not indicators:
                return {}

            # Get current price
            price_data = await self.market_api.get_stock_data(symbol, user_id)
            if not price_data:
                return {}

            current_price = price_data['price']
            
            # Analyze trend
            trend = {
                'direction': self._determine_trend_direction(indicators, current_price),
                'strength': self._calculate_trend_strength(indicators),
                'support_levels': indicators.get('support_resistance', {}).get('support', []),
                'resistance_levels': indicators.get('support_resistance', {}).get('resistance', []),
                'timestamp': datetime.now()
            }
            
            return trend
        except Exception as e:
            logger.error(f"Error analyzing market trend: {e}")
            return {}

    def _determine_trend_direction(self, indicators: Dict, current_price: float) -> str:
        """Determine trend direction"""
        try:
            ma_20 = indicators['moving_averages']['ma_20']
            ma_50 = indicators['moving_averages']['ma_50']
            ma_200 = indicators['moving_averages']['ma_200']
            
            if not all([ma_20, ma_50, ma_200]):
                return 'neutral'
            
            if current_price > ma_20 > ma_50 > ma_200:
                return 'strong_uptrend'
            elif current_price > ma_20 > ma_50:
                return 'uptrend'
            elif current_price < ma_20 < ma_50 < ma_200:
                return 'strong_downtrend'
            elif current_price < ma_20 < ma_50:
                return 'downtrend'
            else:
                return 'neutral'
        except Exception as e:
            logger.error(f"Error determining trend direction: {e}")
            return 'neutral'

    def _calculate_trend_strength(self, indicators: Dict) -> float:
        """Calculate trend strength"""
        try:
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', {}).get('histogram', 0)
            
            # Normalize RSI to -1 to 1 range
            rsi_strength = (rsi - 50) / 50
            
            # Normalize MACD (assuming typical range)
            macd_strength = min(max(macd / 2, -1), 1)
            
            # Combine indicators
            strength = (rsi_strength + macd_strength) / 2
            
            return min(max(strength, -1), 1)
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0

    async def get_market_scanner_results(self, filters: Dict, user_id: int) -> List[Dict]:
        """Get market scanner results based on filters"""
        try:
            # Get all symbols
            symbols = await self._get_all_symbols()
            results = []
            
            # Scan each symbol
            for symbol in symbols:
                analysis = await self.get_stock_analysis(symbol, user_id)
                if self._matches_filters(analysis, filters):
                    results.append(analysis)
            
            return results
        except Exception as e:
            logger.error(f"Error getting market scanner results: {e}")
            return []

    async def _get_all_symbols(self) -> List[str]:
        """Get all available symbols"""
        try:
            # Get symbols from database
            symbols = await self.db.get_all_symbols()
            return symbols
        except Exception as e:
            logger.error(f"Error getting all symbols: {e}")
            return []

    def _matches_filters(self, analysis: Dict, filters: Dict) -> bool:
        """Check if analysis matches filters"""
        try:
            if not analysis:
                return False
            
            # Check price filters
            if 'min_price' in filters and analysis['price_data']['price'] < filters['min_price']:
                return False
            if 'max_price' in filters and analysis['price_data']['price'] > filters['max_price']:
                return False
            
            # Check volume filters
            if 'min_volume' in filters and analysis['price_data']['volume'] < filters['min_volume']:
                return False
            
            # Check technical indicator filters
            indicators = analysis['technical_indicators']
            if 'min_rsi' in filters and indicators['rsi'] < filters['min_rsi']:
                return False
            if 'max_rsi' in filters and indicators['rsi'] > filters['max_rsi']:
                return False
            
            # Check trend filters
            if 'trend' in filters and analysis['trend']['direction'] != filters['trend']:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error matching filters: {e}")
            return False 