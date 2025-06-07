import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')
FIXER_API_KEY = os.getenv('FIXER_API_KEY')
GOOGLE_AI_API_KEY = os.getenv('GOOGLE_AI_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Database Configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
DB_NAME = os.getenv('DB_NAME', 'tradeflow_bot')

# Bot Settings
BOT_OWNER_ID = int(os.getenv('BOT_OWNER_ID', '0'))
DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', 'en')
SUPPORT_CHAT_ID = os.getenv('SUPPORT_CHAT_ID')

# Subscription Settings
SUBSCRIPTION_PLANS = {
    'basic': {
        'name': 'Basic',
        'price': 9.99,
        'features': [
            'Basic market data',
            'Price alerts',
            'Portfolio tracking'
        ]
    },
    'pro': {
        'name': 'Professional',
        'price': 29.99,
        'features': [
            'Advanced market data',
            'Technical analysis',
            'ORB trading',
            'AI trading chat',
            'Priority support'
        ]
    },
    'enterprise': {
        'name': 'Enterprise',
        'price': 99.99,
        'features': [
            'All Pro features',
            'Custom indicators',
            'API access',
            'Dedicated support',
            'White-label options'
        ]
    }
}

# Trading Settings
DEFAULT_TIMEFRAME = '1d'
DEFAULT_SYMBOL = 'SPY'
MIN_ALERT_PRICE = 0.01
MAX_ALERT_PRICE = 1000000
MIN_POSITION_SIZE = 1
MAX_POSITION_SIZE = 1000000

# Affiliate Program Settings
AFFILIATE_COMMISSION_RATE = 0.20  # 20% commission
MIN_PAYOUT_AMOUNT = 50  # Minimum $50 for payout
REFERRAL_BONUS = 10  # $10 bonus for referred users

# AI Chat Settings
AI_MODEL = 'gemini-pro'
AI_TEMPERATURE = 0.7
AI_MAX_TOKENS = 1000
AI_CHAT_HISTORY_LIMIT = 10

# Market Data Settings
MARKET_DATA_REFRESH_INTERVAL = 60  # seconds
PRICE_ALERT_CHECK_INTERVAL = 30  # seconds
TECHNICAL_INDICATORS = [
    'RSI',
    'MACD',
    'SMA',
    'EMA',
    'Bollinger Bands'
]

# Error Messages
ERROR_MESSAGES = {
    'api_error': 'Error fetching data from API. Please try again later.',
    'db_error': 'Database error occurred. Please try again.',
    'auth_error': 'Authentication error. Please check your credentials.',
    'subscription_required': 'This feature requires an active subscription.',
    'invalid_input': 'Invalid input provided. Please check and try again.',
    'rate_limit': 'Rate limit exceeded. Please try again later.'
}

# Success Messages
SUCCESS_MESSAGES = {
    'subscription_activated': 'Subscription activated successfully!',
    'alert_created': 'Price alert created successfully!',
    'position_added': 'Position added to portfolio successfully!',
    'payout_requested': 'Payout request submitted successfully!',
    'referral_processed': 'Referral processed successfully!'
}

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'tradeflow_bot.log'

# Cache Settings
CACHE_TTL = 300  # 5 minutes
CACHE_MAX_SIZE = 1000

# Security Settings
API_RATE_LIMIT = 100  # requests per minute
MAX_LOGIN_ATTEMPTS = 3
SESSION_TIMEOUT = 3600  # 1 hour
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION = 3600  # 1 hour

# Feature Flags
FEATURES = {
    'ai_chat': True,
    'market_scanner': True,
    'orb_trading': True,
    'affiliate_program': True,
    'trading_academy': True
}

# Notification Settings
NOTIFICATION_CHANNELS = {
    'telegram': True,
    'email': False,
    'push': False
}

# Timezone Settings
DEFAULT_TIMEZONE = 'UTC'
MARKET_HOURS = {
    'US': {
        'open': '09:30',
        'close': '16:00',
        'timezone': 'America/New_York'
    },
    'CRYPTO': {
        'open': '00:00',
        'close': '23:59',
        'timezone': 'UTC'
    }
}

# AI Chat Settings
AI_CHAT_SETTINGS = {
    'model': AI_MODEL,
    'temperature': AI_TEMPERATURE,
    'max_tokens': AI_MAX_TOKENS,
    'history_limit': AI_CHAT_HISTORY_LIMIT
} 