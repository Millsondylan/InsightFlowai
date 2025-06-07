import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
from config import (
    ALPHA_VANTAGE_API_KEY,
    COINGECKO_API_KEY,
    FIXER_API_KEY,
    NEWS_API_KEY,
    MARKET_DATA_REFRESH_INTERVAL,
    API_RATE_LIMIT
)

logger = logging.getLogger(__name__)

class MarketDataAPI:
    """Handles market data API calls"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_timestamps = {}
        self.rate_limits = {}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _check_rate_limit(self, endpoint: str, user_id: int) -> bool:
        """Check if API rate limit is exceeded"""
        key = f"{endpoint}:{user_id}"
        now = datetime.now()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {
                'count': 0,
                'window_start': now
            }
        
        if now - self.rate_limits[key]['window_start'] > timedelta(minutes=1):
            self.rate_limits[key] = {
                'count': 0,
                'window_start': now
            }
        
        if self.rate_limits[key]['count'] >= API_RATE_LIMIT:
            return False
        
        self.rate_limits[key]['count'] += 1
        return True

    async def _get_cached_data(self, key: str) -> Optional[Dict]:
        """Get cached data if available and not expired"""
        if key in self.cache:
            if datetime.now() - self.cache_timestamps[key] < timedelta(seconds=MARKET_DATA_REFRESH_INTERVAL):
                return self.cache[key]
        return None

    async def _cache_data(self, key: str, data: Dict):
        """Cache data with timestamp"""
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now()

    async def get_stock_data(self, symbol: str, user_id: int) -> Optional[Dict]:
        """Get stock data from Alpha Vantage"""
        try:
            if not await self._check_rate_limit('stock', user_id):
                raise Exception("Rate limit exceeded")

            cache_key = f"stock:{symbol}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data

            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
            async with self.session.get(url) as response:
                data = await response.json()
                
                if 'Global Quote' in data:
                    quote = data['Global Quote']
                    result = {
                        'symbol': symbol,
                        'price': float(quote['05. price']),
                        'change': float(quote['09. change']),
                        'change_percent': float(quote['10. change percent'].strip('%')),
                        'volume': int(quote['06. volume']),
                        'timestamp': datetime.now()
                    }
                    await self._cache_data(cache_key, result)
                    return result
                return None
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            return None

    async def get_crypto_data(self, symbol: str, user_id: int) -> Optional[Dict]:
        """Get cryptocurrency data from CoinGecko"""
        try:
            if not await self._check_rate_limit('crypto', user_id):
                raise Exception("Rate limit exceeded")

            cache_key = f"crypto:{symbol}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data

            url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
            async with self.session.get(url) as response:
                data = await response.json()
                
                if symbol in data:
                    result = {
                        'symbol': symbol,
                        'price': data[symbol]['usd'],
                        'change': data[symbol]['usd_24h_change'],
                        'volume': data[symbol]['usd_24h_vol'],
                        'timestamp': datetime.now()
                    }
                    await self._cache_data(cache_key, result)
                    return result
                return None
        except Exception as e:
            logger.error(f"Error fetching crypto data: {e}")
            return None

    async def get_forex_data(self, pair: str, user_id: int) -> Optional[Dict]:
        """Get forex data from Fixer"""
        try:
            if not await self._check_rate_limit('forex', user_id):
                raise Exception("Rate limit exceeded")

            cache_key = f"forex:{pair}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data

            url = f"http://data.fixer.io/api/latest?access_key={FIXER_API_KEY}&base=EUR&symbols={pair}"
            async with self.session.get(url) as response:
                data = await response.json()
                
                if 'rates' in data:
                    result = {
                        'pair': pair,
                        'rate': data['rates'][pair],
                        'timestamp': datetime.now()
                    }
                    await self._cache_data(cache_key, result)
                    return result
                return None
        except Exception as e:
            logger.error(f"Error fetching forex data: {e}")
            return None

    async def get_market_news(self, user_id: int) -> List[Dict]:
        """Get market news from News API"""
        try:
            if not await self._check_rate_limit('news', user_id):
                raise Exception("Rate limit exceeded")

            cache_key = "market_news"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data

            url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey={NEWS_API_KEY}"
            async with self.session.get(url) as response:
                data = await response.json()
                
                if 'articles' in data:
                    news = [{
                        'title': article['title'],
                        'description': article['description'],
                        'url': article['url'],
                        'published_at': article['publishedAt']
                    } for article in data['articles']]
                    await self._cache_data(cache_key, news)
                    return news
                return []
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []

    async def get_technical_indicators(self, symbol: str, indicator: str, timeframe: str, user_id: int) -> Optional[Dict]:
        """Get technical indicators from Alpha Vantage"""
        try:
            if not await self._check_rate_limit('technical', user_id):
                raise Exception("Rate limit exceeded")

            cache_key = f"technical:{symbol}:{indicator}:{timeframe}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data

            url = f"https://www.alphavantage.co/query?function={indicator}&symbol={symbol}&interval={timeframe}&apikey={ALPHA_VANTAGE_API_KEY}"
            async with self.session.get(url) as response:
                data = await response.json()
                
                if indicator in data:
                    result = {
                        'symbol': symbol,
                        'indicator': indicator,
                        'value': float(data[indicator][0][indicator]),
                        'timestamp': datetime.now(),
                        'timeframe': timeframe
                    }
                    await self._cache_data(cache_key, result)
                    return result
                return None
        except Exception as e:
            logger.error(f"Error fetching technical indicators: {e}")
            return None

    async def get_market_overview(self, user_id: int) -> Dict:
        """Get comprehensive market overview"""
        try:
            if not await self._check_rate_limit('overview', user_id):
                raise Exception("Rate limit exceeded")

            cache_key = "market_overview"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data

            # Fetch data for major indices
            indices = ['SPY', 'QQQ', 'DIA']
            stocks_data = await asyncio.gather(*[
                self.get_stock_data(symbol, user_id) for symbol in indices
            ])

            # Fetch data for major cryptocurrencies
            cryptos = ['bitcoin', 'ethereum']
            crypto_data = await asyncio.gather(*[
                self.get_crypto_data(symbol, user_id) for symbol in cryptos
            ])

            # Fetch forex data
            forex_pairs = ['USD', 'EUR', 'GBP']
            forex_data = await asyncio.gather(*[
                self.get_forex_data(pair, user_id) for pair in forex_pairs
            ])

            # Fetch market news
            news = await self.get_market_news(user_id)

            result = {
                'stocks': {data['symbol']: data for data in stocks_data if data},
                'crypto': {data['symbol']: data for data in crypto_data if data},
                'forex': {data['pair']: data for data in forex_data if data},
                'news': news,
                'timestamp': datetime.now()
            }
            await self._cache_data(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error fetching market overview: {e}")
            return {} 