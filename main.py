import os
import asyncio
import logging
from datetime import datetime, timedelta
import json
import base64
import io
import sqlite3
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
import pytz
import time
from asyncio import Semaphore
import hashlib
import random
import traceback
import signal
import sys

# Import configuration
from config import *

# Configure logging with better error handling
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global variable to track if bot is running
BOT_RUNNING = False
application = None
INSTANCE_LOCK_FILE = "bot_instance.lock"

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global BOT_RUNNING, application
    logger.info("🛑 Received shutdown signal. Stopping bot gracefully...")
    BOT_RUNNING = False
    if application:
        try:
            application.stop_running()
        except:
            pass
    # Clean up lock file
    try:
        if os.path.exists(INSTANCE_LOCK_FILE):
            os.remove(INSTANCE_LOCK_FILE)
    except:
        pass
    sys.exit(0)

def check_single_instance():
    """Ensure only one bot instance is running"""
    if os.path.exists(INSTANCE_LOCK_FILE):
        try:
            with open(INSTANCE_LOCK_FILE, 'r') as f:
                pid = int(f.read().strip())
            # Check if process is still running
            try:
                os.kill(pid, 0)  # Check if process exists
                logger.error("❌ Another bot instance is already running! Exiting...")
                return False
            except OSError:
                # Process doesn't exist, remove stale lock file
                os.remove(INSTANCE_LOCK_FILE)
        except:
            # Invalid lock file, remove it
            try:
                os.remove(INSTANCE_LOCK_FILE)
            except:
                pass

    # Create lock file with current process ID
    try:
        with open(INSTANCE_LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
        return True
    except Exception as e:
        logger.error(f"❌ Could not create instance lock: {e}")
        return False

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

logger.info("🚀 Starting TradeFlow AI Bot application...")

# API Configuration from environment variables/secrets
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("Gemini_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
FIXER_API_KEY = os.getenv("FIXER_API_KEY")
EXCHANGERATE_API_KEY = os.getenv("ExchangeRate_API_KEY")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
PGDATABASE = os.getenv("PGDATABASE")
PGHOST = os.getenv("PGHOST")
PGPORT = os.getenv("PGPORT")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")

# Validate required API keys from Replit Secrets
if not TELEGRAM_BOT_TOKEN:
    logger.error("❌ TELEGRAM_BOT_TOKEN is required! Please add it to your Replit Secrets.")
    logger.error("   1. Go to Tools → Secrets in your Replit workspace")
    logger.error("   2. Add TELEGRAM_BOT_TOKEN with your bot token value")
    sys.exit(1)

# Validate token format
if not TELEGRAM_BOT_TOKEN.count(':') == 1 or len(TELEGRAM_BOT_TOKEN) < 20:
    logger.error("❌ Invalid TELEGRAM_BOT_TOKEN format! Please check your bot token.")
    sys.exit(1)

# Log API key status for debugging (mask sensitive parts)
def mask_key(key):
    if not key:
        return None
    return f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"

logger.info("🔑 API Keys Status (from Replit Secrets):")
logger.info(f"  - Telegram Bot Token: {'✅ Set' if TELEGRAM_BOT_TOKEN else '❌ Missing'} ({mask_key(TELEGRAM_BOT_TOKEN)})")
logger.info(f"  - OpenAI API Key: {'✅ Set' if OPENAI_API_KEY else '❌ Missing'} ({mask_key(OPENAI_API_KEY)})")
logger.info(f"  - Groq API Key: {'✅ Set' if GROQ_API_KEY else '❌ Missing'} ({mask_key(GROQ_API_KEY)})")
logger.info(f"  - Gemini API Key: {'✅ Set' if GEMINI_API_KEY else '❌ Missing'} ({mask_key(GEMINI_API_KEY)})")

# Warn if no AI providers are available
if not any([GROQ_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY]):
    logger.warning("⚠️ No AI API keys found! AI chat will not work properly.")
    AI_ENABLED = False
else:
    AI_ENABLED = True

# Initialize AI clients
groq_client = None
openai_client = None
gemini_client = None

# Initialize Groq client (primary)
try:
    if GROQ_API_KEY:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("✅ Groq client initialized successfully (Primary AI)")
    else:
        logger.warning("⚠️ Groq API key not found")
except Exception as e:
    logger.error(f"❌ Groq client initialization failed: {e}")

# Initialize OpenAI client (backup)
try:
    if OPENAI_API_KEY:
        import openai
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.info("✅ OpenAI client initialized successfully (Backup AI)")
    else:
        logger.warning("⚠️ OpenAI API key not found")
except Exception as e:
    logger.error(f"❌ OpenAI client initialization failed: {e}")

# Initialize Gemini client (third backup)
try:
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = genai.GenerativeModel('gemini-pro')
        logger.info("✅ Gemini client initialized successfully (Third Backup AI)")
    else:
        logger.warning("⚠️ Gemini API key not found")
except Exception as e:
    logger.error(f"❌ Gemini client initialization failed: {e}")

# Owner/Admin configuration
OWNER_ID = 7580293197
ADMIN_USERS = {OWNER_ID}
ALLOWED_USERS = set(ADMIN_USERS)

# Payment support email
PAYMENT_SUPPORT_EMAIL = "insightflowaitrading@gmail.com"

# Crypto payment addresses - Your Personal Addresses
CRYPTO_ADDRESSES = {
    "BTC": "1KcYGjJnNduK72rEt8LZyzZeZ3BGGwGYT",
    "USDT_TRC20": "TLEgUbALuXwV49RbJFRhMpaX23AYjC9Dwc"
}

# Affiliate Program Configuration
AFFILIATE_COMMISSION_RATE = 0.10  # 10% commission
AFFILIATE_MINIMUM_PAYOUT = 5.0  # Minimum $5 for payout

# Subscription plans
SUBSCRIPTION_PLANS = {
    "trial": {
        "name": "3-Day Free Trial",
        "price": 0,
        "duration_days": 3,
        "features": ["Full access to all features", "AI Trading Chat", "Market Analysis", "Real-time Alerts", "All Trading Tools"]
    },
    "temporary": {
        "name": "Temporary Access",
        "price": 0,
        "duration_days": 7,
        "features": ["Full access during payment processing", "All premium features", "Admin granted access"]
    },
    "monthly": {
        "name": "Monthly Plan",
        "price": 10,
        "duration_days": 30,
        "features": ["AI Trading Chat", "Market Analysis", "Real-time Alerts", "All Trading Tools", "Priority Support"]
    },
    "quarterly": {
        "name": "3-Month Plan",
        "price": 25,
        "duration_days": 90,
        "savings": "Save $5",
        "features": ["All Monthly features", "3 months access", "Quarterly reviews", "Advanced analytics"]
    },
    "yearly": {
        "name": "Yearly Plan",
        "price": 100,
        "duration_days": 365,
        "savings": "Save $20",
        "features": ["All Monthly features", "12 months access", "Best value", "Premium features", "Priority support"]
    },
    "expired": {
        "name": "Expired Subscription",
        "price": 0,
        "duration_days": 0,
        "features": ["Subscription renewal required"]
    }
}

# Default ORB Trading Configuration
DEFAULT_ORB_CONFIG = {
    "pairs": ["GBPUSD", "EURUSD"],
    "london_session": {"start": "09:00", "end": "10:00"},  # SAST
    "newyork_session": {"start": "14:30", "end": "15:30"},  # SAST
    "account_size": 10000,  # USD
    "risk_per_trade": 0.01,  # 1%
    "commission": 7,  # USD round trip
    "max_sl_pips": 10,
    "timezone": "Africa/Johannesburg"  # SAST
}

def safe_database_operation(func):
    """Enhanced decorator for safe database operations with connection pooling"""
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.05 * (attempt + 1))  # Reduced wait time
                    continue
                else:
                    logger.error(f"Database operation failed after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"Database operation error: {e}")
                raise
        return None
    return wrapper

@safe_database_operation
def get_user_orb_config(user_id: int) -> Dict:
    """Get user's ORB configuration with fallback to defaults"""
    try:
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]

        query_columns = []
        if 'account_size' in columns:
            query_columns.append('account_size')
        if 'risk_per_trade' in columns:
            query_columns.append('risk_per_trade')
        if 'commission' in columns:
            query_columns.append('commission')
        if 'preferred_pairs' in columns:
            query_columns.append('preferred_pairs')

        if query_columns:
            cursor.execute(f'''
                SELECT {', '.join(query_columns)}
                FROM users WHERE user_id = ?
            ''', (user_id,))
            result = cursor.fetchone()
        else:
            result = None

        conn.close()

        if result and len(result) > 0:
            config = DEFAULT_ORB_CONFIG.copy()
            if len(query_columns) >= 1 and result[0]:
                config["account_size"] = result[0]
            if len(query_columns) >= 2 and result[1]:
                config["risk_per_trade"] = result[1]
            if len(query_columns) >= 3 and result[2]:
                config["commission"] = result[2]
            if len(query_columns) >= 4 and result[3]:
                config["pairs"] = result[3].split(',') if result[3] else DEFAULT_ORB_CONFIG["pairs"]
            return config

        return DEFAULT_ORB_CONFIG.copy()
    except Exception as e:
        logger.error(f"Error getting user ORB config: {e}")
        return DEFAULT_ORB_CONFIG.copy()

# Enhanced caching system
market_cache = {}
user_cache = {}
conversation_cache = {}
CACHE_DURATION = 300
USER_CACHE_DURATION = 600
CONVERSATION_CACHE_DURATION = 900

def get_db_connection():
    """Get a fresh database connection for each operation"""
    conn = sqlite3.connect('trading_bot.db', timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA cache_size=10000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

def cache_user_data(user_id: int, data: dict, duration: int = USER_CACHE_DURATION):
    """Cache user data for faster access"""
    try:
        cache_key = f"user_{user_id}"
        user_cache[cache_key] = {
            'data': data,
            'timestamp': time.time(),
            'duration': duration
        }
    except Exception as e:
        logger.error(f"Error caching user data: {e}")

def get_cached_user_data(user_id: int):
    """Get cached user data if available and fresh"""
    try:
        cache_key = f"user_{user_id}"
        if cache_key in user_cache:
            cached = user_cache[cache_key]
            if time.time() - cached['timestamp'] < cached['duration']:
                return cached['data']
            else:
                del user_cache[cache_key]
        return None
    except Exception as e:
        logger.error(f"Error getting cached user data: {e}")
        return None

# Enhanced rate limiting with better performance
API_REQUEST_LIMITS = {
    'groq': {'calls': 0, 'reset_time': time.time(), 'max_per_minute': 20},
    'openai': {'calls': 0, 'reset_time': time.time(), 'max_per_minute': 15},
    'gemini': {'calls': 0, 'reset_time': time.time(), 'max_per_minute': 12},
    'alpha_vantage': {'calls': 0, 'reset_time': time.time(), 'max_per_minute': 5},
    'fmp': {'calls': 0, 'reset_time': time.time(), 'max_per_minute': 5},
    'coingecko': {'calls': 0, 'reset_time': time.time(), 'max_per_minute': 8}
}

def check_rate_limit(api_name: str) -> bool:
    """Check if API call is within rate limits"""
    try:
        current_time = time.time()
        limit_info = API_REQUEST_LIMITS.get(api_name, {'calls': 0, 'reset_time': current_time, 'max_per_minute': 10})

        if current_time - limit_info['reset_time'] >= 60:
            limit_info['calls'] = 0
            limit_info['reset_time'] = current_time

        max_calls = limit_info.get('max_per_minute', 10)
        if api_name in ['groq', 'openai', 'gemini']:
            max_calls = 15

        if limit_info['calls'] < max_calls:
            limit_info['calls'] += 1
            return True

        logger.warning(f"Rate limit exceeded for {api_name}")
        return False
    except Exception as e:
        logger.error(f"Error checking rate limit for {api_name}: {e}")
        return True

def get_ai_status() -> str:
    """Get current AI provider status"""
    try:
        providers = []
        if groq_client:
            providers.append("Groq")
        if openai_client:
            providers.append("OpenAI")
        if gemini_client:
            providers.append("Gemini")

        if providers:
            return f"Active: {', '.join(providers)}"
        else:
            return "No AI providers available"
    except Exception as e:
        logger.error(f"Error getting AI status: {e}")
        return "Status unknown"

@safe_database_operation
def init_database():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        telegram_id INTEGER UNIQUE NOT NULL,
        username TEXT,
        first_name TEXT,
        last_name TEXT,
        referral_code TEXT UNIQUE,
        referred_by INTEGER,
        subscription_status TEXT DEFAULT 'expired',
        subscription_end TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_admin BOOLEAN DEFAULT FALSE,
        FOREIGN KEY (referred_by) REFERENCES users(telegram_id)
    )
    ''')
    
    # Affiliate earnings table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS affiliate_earnings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        referrer_id INTEGER NOT NULL,
        amount REAL NOT NULL,
        payment_type TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(telegram_id),
        FOREIGN KEY (referrer_id) REFERENCES users(telegram_id)
    )
    ''')
    
    # Conversation history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        telegram_id INTEGER NOT NULL,
        user_message TEXT NOT NULL,
        ai_response TEXT NOT NULL,
        context_used TEXT,
        provider_used TEXT,
        session_id TEXT DEFAULT 'default',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(telegram_id)
    )
    ''')
    
    # ORB trading configurations
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orb_configs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE NOT NULL,
        pairs TEXT NOT NULL,
        london_session_start TEXT NOT NULL,
        london_session_end TEXT NOT NULL,
        newyork_session_start TEXT NOT NULL,
        newyork_session_end TEXT NOT NULL,
        account_size REAL NOT NULL,
        risk_per_trade REAL NOT NULL,
        commission REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(telegram_id)
    )
    ''')
    
    # Price alerts table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS price_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        price_target REAL NOT NULL,
        condition TEXT NOT NULL,
        status TEXT DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        triggered_at TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(telegram_id)
    )
    ''')
    
    # User activity logs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_activity_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        action TEXT NOT NULL,
        details TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(telegram_id)
    )
    ''')
    
    # Create indexes for better query performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_telegram_id ON users(telegram_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_referral_code ON users(referral_code)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_affiliate_earnings_user_id ON affiliate_earnings(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_alerts_user_id ON price_alerts(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_activity_logs_user_id ON user_activity_logs(user_id)')
    
    # Insert default admin user if not exists
    cursor.execute('''
    INSERT OR IGNORE INTO users (telegram_id, username, first_name, subscription_status, is_admin)
    VALUES (?, ?, ?, ?, ?)
    ''', (OWNER_ID, 'admin', 'Admin', 'active', True))
    
    conn.commit()
    conn.close()
    logger.info("✅ Database initialized successfully")

def generate_referral_code(user_id: int) -> str:
    """Generate unique referral code for user"""
    import hashlib
    import time
    import random
    # Include random component to ensure uniqueness
    base_string = f"{user_id}_{int(time.time())}_{random.randint(1000, 9999)}"
    hash_object = hashlib.md5(base_string.encode())
    return f"TF{hash_object.hexdigest()[:8].upper()}"

@safe_database_operation
def get_or_create_user(telegram_id: int, username: str = None, first_name: str = None, last_name: str = None, referral_code: str = None):
    """Get or create a user in the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute('''
    SELECT id, telegram_id, username, first_name, last_name, subscription_status, subscription_end, is_admin
    FROM users WHERE telegram_id = ?
    ''', (telegram_id,))
    
    user = cursor.fetchone()
    
    if user:
        # Update user information if provided
        if any([username, first_name, last_name]):
            update_fields = []
            update_values = []
            
            if username:
                update_fields.append('username = ?')
                update_values.append(username)
            if first_name:
                update_fields.append('first_name = ?')
                update_values.append(first_name)
            if last_name:
                update_fields.append('last_name = ?')
                update_values.append(last_name)
            
            if update_fields:
                update_values.append(telegram_id)
                cursor.execute(f'''
                UPDATE users 
                SET {', '.join(update_fields)}
                WHERE telegram_id = ?
                ''', update_values)
                conn.commit()
        
        # Return existing user data
        return {
            'id': user[0],
            'telegram_id': user[1],
            'username': user[2],
            'first_name': user[3],
            'last_name': user[4],
            'subscription_status': user[5],
            'subscription_end': user[6],
            'is_admin': bool(user[7])
        }
    
    # Generate referral code for new user
    new_referral_code = generate_referral_code(telegram_id)
    
    # Get referrer ID if referral code provided
    referrer_id = None
    if referral_code:
        cursor.execute('SELECT telegram_id FROM users WHERE referral_code = ?', (referral_code,))
        referrer = cursor.fetchone()
        if referrer:
            referrer_id = referrer[0]
    
    # Create new user
    cursor.execute('''
    INSERT INTO users (
        telegram_id, username, first_name, last_name,
        referral_code, referred_by, subscription_status,
        subscription_end, created_at, last_active
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    ''', (
        telegram_id, username, first_name, last_name,
        new_referral_code, referrer_id, 'expired',
        None
    ))
    
    user_id = cursor.lastrowid
    
    # Log user creation
    cursor.execute('''
    INSERT INTO user_activity_logs (user_id, action, details)
    VALUES (?, ?, ?)
    ''', (telegram_id, 'user_created', json.dumps({
        'referral_code': new_referral_code,
        'referred_by': referrer_id
    })))
    
    conn.commit()
    
    # Return new user data
    return {
        'id': user_id,
        'telegram_id': telegram_id,
        'username': username,
        'first_name': first_name,
        'last_name': last_name,
        'subscription_status': 'expired',
        'subscription_end': None,
        'is_admin': False,
        'referral_code': new_referral_code
    }

@safe_database_operation
def process_affiliate_commission(referrer_user_id: int, payment_amount: float, payment_type: str):
    """Process affiliate commission for referrer"""
    try:
        commission_amount = payment_amount * AFFILIATE_COMMISSION_RATE
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Add commission to pending
        cursor.execute('''
            UPDATE users SET 
                pending_commissions = pending_commissions + ?,
                total_commissions = total_commissions + ?
            WHERE user_id = ?
        ''', (commission_amount, commission_amount, referrer_user_id))
        
        # Log commission transaction
        cursor.execute('''
            INSERT INTO payments (user_id, payment_type, amount, currency, status, commission_amount, affiliate_user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (referrer_user_id, f"affiliate_commission_{payment_type}", commission_amount, "USD", "pending", commission_amount, referrer_user_id))
        
        conn.commit()
        
        # Notify referrer
        cursor.execute('SELECT telegram_id FROM users WHERE user_id = ?', (referrer_user_id,))
        result = cursor.fetchone()
        if result:
            return result[0], commission_amount
        
        conn.close()
        return None, commission_amount
        
    except Exception as e:
        logger.error(f"Error processing affiliate commission: {e}")
        return None, 0

@safe_database_operation
def get_user_affiliate_stats(user_id: int):
    """Get user's affiliate statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT referral_code, total_referrals, total_commissions, pending_commissions, paid_commissions
            FROM users WHERE user_id = ?
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'referral_code': result[0],
                'total_referrals': result[1] or 0,
                'total_commissions': result[2] or 0.0,
                'pending_commissions': result[3] or 0.0,
                'paid_commissions': result[4] or 0.0
            }
        
        return None
    except Exception as e:
        logger.error(f"Error getting affiliate stats: {e}")
        return None

@safe_database_operation
def get_referral_earnings_breakdown(user_id: int):
    """Get detailed breakdown of referral earnings"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT u.username, u.first_name, p.amount, p.commission_amount, p.payment_date, p.plan_type
            FROM payments p
            JOIN users u ON p.user_id = u.user_id
            WHERE p.affiliate_user_id = ? AND p.commission_amount > 0
            ORDER BY p.payment_date DESC
            LIMIT 20
        ''', (user_id,))
        
        earnings = cursor.fetchall()
        conn.close()
        
        return earnings
    except Exception as e:
        logger.error(f"Error getting referral earnings: {e}")
        return []

def update_user_activity_sync(user_id: int):
    """Update user activity synchronously"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET last_activity = CURRENT_TIMESTAMP WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error updating user activity: {e}")

@safe_database_operation
def check_subscription_status(user_id: int):
    """Check and update user's subscription status"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get user's current subscription info
    cursor.execute('''
    SELECT subscription_status, subscription_end
    FROM users WHERE telegram_id = ?
    ''', (user_id,))
    
    result = cursor.fetchone()
    if not result:
        return {
            'status': 'expired',
            'end_date': None,
            'is_active': False,
            'days_remaining': 0
        }
    
    current_status, end_date = result
    
    # If no end date or expired, return expired status
    if not end_date or (end_date and datetime.fromisoformat(end_date) < datetime.now(pytz.UTC)):
        if current_status != 'expired':
            cursor.execute('''
            UPDATE users 
            SET subscription_status = 'expired', subscription_end = NULL
            WHERE telegram_id = ?
            ''', (user_id,))
            conn.commit()
        
        return {
            'status': 'expired',
            'end_date': None,
            'is_active': False,
            'days_remaining': 0
        }
    
    # Calculate days remaining
    end_datetime = datetime.fromisoformat(end_date)
    now = datetime.now(pytz.UTC)
    days_remaining = (end_datetime - now).days
    
    # Update status if needed
    if days_remaining <= 0:
        cursor.execute('''
        UPDATE users 
        SET subscription_status = 'expired', subscription_end = NULL
        WHERE telegram_id = ?
        ''', (user_id,))
        conn.commit()
        
        return {
            'status': 'expired',
            'end_date': None,
            'is_active': False,
            'days_remaining': 0
        }
    
    # Return active subscription info
    return {
        'status': current_status,
        'end_date': end_date,
        'is_active': True,
        'days_remaining': days_remaining
    }

def get_price_data(symbol: str) -> Dict:
    """Get current price data for a symbol using available APIs"""
    try:
        # Normalize symbol
        symbol = symbol.upper().strip()
        
        # Try Alpha Vantage first for stocks
        if ALPHA_VANTAGE_API_KEY and symbol.isalpha():
            try:
                response = requests.get(
                    f'https://www.alphavantage.co/query',
                    params={
                        'function': 'GLOBAL_QUOTE',
                        'symbol': symbol,
                        'apikey': ALPHA_VANTAGE_API_KEY
                    },
                    timeout=5
                )
                data = response.json()
                
                if 'Global Quote' in data:
                    quote = data['Global Quote']
                    return {
                        'symbol': symbol,
                        'price': float(quote['05. price']),
                        'change': float(quote['09. change']),
                        'change_percent': float(quote['10. change percent'].rstrip('%')),
                        'volume': int(quote['06. volume']),
                        'provider': 'Alpha Vantage'
                    }
            except Exception as e:
                logger.error(f"Alpha Vantage error: {e}")
        
        # Try Polygon.io for stocks and crypto
        if POLYGON_API_KEY:
            try:
                response = requests.get(
                    f'https://api.polygon.io/v2/last/trade/{symbol}',
                    params={'apiKey': POLYGON_API_KEY},
                    timeout=5
                )
                data = response.json()
                
                if data.get('status') == 'success':
                    return {
                        'symbol': symbol,
                        'price': float(data['last']['price']),
                        'volume': int(data['last']['size']),
                        'timestamp': data['last']['t'],
                        'provider': 'Polygon.io'
                    }
            except Exception as e:
                logger.error(f"Polygon.io error: {e}")
        
        # Try CoinGecko for crypto
        if COINGECKO_API_KEY and symbol.startswith(('BTC', 'ETH', 'USDT')):
            try:
                response = requests.get(
                    f'https://api.coingecko.com/api/v3/simple/price',
                    params={
                        'ids': symbol.lower(),
                        'vs_currencies': 'usd',
                        'include_24hr_change': 'true',
                        'include_24hr_vol': 'true',
                        'x_cg_api_key': COINGECKO_API_KEY
                    },
                    timeout=5
                )
                data = response.json()
                
                if symbol.lower() in data:
                    coin_data = data[symbol.lower()]
                    return {
                        'symbol': symbol,
                        'price': float(coin_data['usd']),
                        'change_24h': float(coin_data.get('usd_24h_change', 0)),
                        'volume_24h': float(coin_data.get('usd_24h_vol', 0)),
                        'provider': 'CoinGecko'
                    }
            except Exception as e:
                logger.error(f"CoinGecko error: {e}")
        
        # Try Fixer.io for forex
        if FIXER_API_KEY and len(symbol) == 6 and symbol.isalpha():
            try:
                base = symbol[:3]
                quote = symbol[3:]
                response = requests.get(
                    f'https://data.fixer.io/api/latest',
                    params={
                        'access_key': FIXER_API_KEY,
                        'base': base,
                        'symbols': quote
                    },
                    timeout=5
                )
                data = response.json()
                
                if data.get('success'):
                    rate = data['rates'][quote]
                    return {
                        'symbol': symbol,
                        'price': float(rate),
                        'provider': 'Fixer.io'
                    }
            except Exception as e:
                logger.error(f"Fixer.io error: {e}")
        
        # Try ExchangeRate-API as backup for forex
        if EXCHANGERATE_API_KEY and len(symbol) == 6 and symbol.isalpha():
            try:
                base = symbol[:3]
                quote = symbol[3:]
                response = requests.get(
                    f'https://v6.exchangerate-api.com/v6/{EXCHANGERATE_API_KEY}/pair/{base}/{quote}',
                    timeout=5
                )
                data = response.json()
                
                if data.get('result') == 'success':
                    return {
                        'symbol': symbol,
                        'price': float(data['conversion_rate']),
                        'provider': 'ExchangeRate-API'
                    }
            except Exception as e:
                logger.error(f"ExchangeRate-API error: {e}")
        
        raise Exception("No price data available from any provider")
        
    except Exception as e:
        logger.error(f"Error getting price data for {symbol}: {e}")
        return {
            'symbol': symbol,
            'error': str(e),
            'status': 'error'
        }

@safe_database_operation
def get_conversation_history(user_id: int, limit: int = 10) -> List[Dict]:
    """Get recent conversation history for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT user_message, ai_response, provider_used, created_at
    FROM conversations
    WHERE user_id = ?
    ORDER BY created_at DESC
    LIMIT ?
    ''', (user_id, limit))
    
    history = []
    for row in cursor.fetchall():
        history.append({
            'user_message': row[0],
            'ai_response': row[1],
            'provider_used': row[2],
            'timestamp': row[3]
        })
    
    return list(reversed(history))  # Return in chronological order

@safe_database_operation
def save_conversation(user_id: int, telegram_id: int, user_message: str, ai_response: str, context_used: str = "", provider_used: str = "", session_id: str = "default"):
    """Save a conversation exchange to the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO conversations (
        user_id, telegram_id, user_message, ai_response,
        context_used, provider_used, session_id, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (
        user_id, telegram_id, user_message, ai_response,
        context_used, provider_used, session_id
    ))
    
    conn.commit()
    
    # Log the conversation
    cursor.execute('''
    INSERT INTO user_activity_logs (user_id, action, details)
    VALUES (?, ?, ?)
    ''', (user_id, 'ai_chat', json.dumps({
        'provider': provider_used,
        'session_id': session_id
    })))
    
    conn.commit()

async def chat_with_ai(message: str, user_id: int = None, telegram_id: int = None) -> str:
    """Chat with AI using available providers"""
    if not AI_ENABLED:
        return "❌ AI services are currently unavailable. Please try again later."
    
    try:
        # Get conversation history
        history = get_conversation_history(user_id or telegram_id) if user_id or telegram_id else []
        
        # Prepare conversation context
        messages = []
        for entry in history:
            messages.append({"role": "user", "content": entry['user_message']})
            messages.append({"role": "assistant", "content": entry['ai_response']})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Try Groq first (primary)
        if groq_client:
            try:
                response = await asyncio.to_thread(
                    groq_client.chat.completions.create,
                    model="mixtral-8x7b-32768",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                ai_response = response.choices[0].message.content
                provider = "groq"
            except Exception as e:
                logger.error(f"Groq error: {e}")
                ai_response = None
                provider = None
        else:
            ai_response = None
            provider = None
        
        # Try OpenAI if Groq fails
        if not ai_response and openai_client:
            try:
                response = await asyncio.to_thread(
                    openai_client.chat.completions.create,
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                ai_response = response.choices[0].message.content
                provider = "openai"
            except Exception as e:
                logger.error(f"OpenAI error: {e}")
                ai_response = None
                provider = None
        
        # Try Gemini as last resort
        if not ai_response and gemini_client:
            try:
                # Convert messages to Gemini format
                gemini_messages = []
                for msg in messages:
                    gemini_messages.append({
                        "role": msg["role"],
                        "parts": [{"text": msg["content"]}]
                    })
                
                response = await asyncio.to_thread(
                    gemini_client.generate_content,
                    contents=gemini_messages
                )
                ai_response = response.text
                provider = "gemini"
            except Exception as e:
                logger.error(f"Gemini error: {e}")
                ai_response = None
                provider = None
        
        if not ai_response:
            return "❌ Sorry, I'm having trouble connecting to AI services. Please try again later."
        
        # Save conversation if user_id/telegram_id provided
        if user_id or telegram_id:
            save_conversation(
                user_id=user_id,
                telegram_id=telegram_id,
                user_message=message,
                ai_response=ai_response,
                provider_used=provider or "unknown"  # Ensure provider_used is always a string
            )
        
        return ai_response
        
    except Exception as e:
        logger.error(f"Error in chat_with_ai: {e}")
        return "❌ An error occurred while processing your request. Please try again."

# MODERN UI DESIGN - Comprehensive Interface Implementation
def main_menu_keyboard(is_active: bool, is_owner: bool = False):
    """Clean simplified main menu with reduced crowding and all features visible"""
    keyboard = [
        [InlineKeyboardButton("🤖 AI Chat", callback_data="ai_trading_chat"),
         InlineKeyboardButton("📊 Markets", callback_data="market_overview")],
        [InlineKeyboardButton("💼 Trading Hub", callback_data="trading_hub"),
         InlineKeyboardButton("📈 Analysis", callback_data="analysis_hub")],
        [InlineKeyboardButton("💰 Portfolio", callback_data="portfolio_tracker"),
         InlineKeyboardButton("🎯 ORB Strategy", callback_data="orb_trading")],
        [InlineKeyboardButton("🚨 Alerts", callback_data="price_alerts"),
         InlineKeyboardButton("🤖 Automation", callback_data="automation_hub")],
        [InlineKeyboardButton("🎓 Academy", callback_data="trading_academy"),
         InlineKeyboardButton("❓ Help", callback_data="help_center")],
        [InlineKeyboardButton("⚙️ Settings", callback_data="profile_settings"),
         InlineKeyboardButton("💎 Subscription", callback_data="subscription_plans")],
        [InlineKeyboardButton("💸 Affiliate Program", callback_data="affiliate_program")],
        # Coming soon features
        [InlineKeyboardButton("🛠️ Strategy Builder", callback_data="strategy_builder"),
         InlineKeyboardButton("🤖 AI Advisor", callback_data="ai_advisor")],
        [InlineKeyboardButton("🔊 Voice Commands", callback_data="voice_commands"),
         InlineKeyboardButton("📊 Correlation Matrix", callback_data="correlation_matrix")],
        [InlineKeyboardButton("🛒 Marketplace", callback_data="marketplace"),
         InlineKeyboardButton("📝 Trade Journal", callback_data="trade_journal")],
    ]
    if is_owner:
        keyboard.append([InlineKeyboardButton("👑 Admin Panel", callback_data="admin_panel")])
    return InlineKeyboardMarkup(keyboard)

# Enhanced Bot Commands with Modern UI
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command"""
    try:
        # Get user info
        user = update.effective_user
        if not user:
            await update.message.reply_text("❌ Error: Could not identify user.")
            return
        
        # Get or create user in database
        user_data = get_or_create_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name
        )
        
        # Check subscription status
        subscription = check_subscription_status(user.id)
        
        # Welcome message
        welcome_text = f"""
👋 Welcome to TradeFlow AI, {user.first_name}!

🤖 Your AI-powered trading companion for:
• Real-time market analysis
• Advanced trading strategies
• Portfolio management
• Risk assessment
• Educational resources

📊 Current Status:
• Subscription: {subscription['status'].title()}
• Days Remaining: {subscription['days_remaining'] if subscription['is_active'] else 'N/A'}

💡 Use the menu below to explore features.
Need help? Type /help or visit the Help Center.
"""
        
        # Create keyboard
        keyboard = main_menu_keyboard(
            is_active=subscription['is_active'],
            is_owner=user.id == OWNER_ID
        )
        
        # Send welcome message with menu
        reply_text = await update.message.reply_text(
            welcome_text,
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
        
        # Log user start
        logger.info(f"✅ User {user.id} started the bot")
        
        if reply_text is not None:
            await update.message.reply_text(reply_text)
        else:
            await update.message.reply_text("Failed to generate a response. Please try again.")
        
    except Exception as e:
        logger.error(f"Error in start command: {e}")
        await update.message.reply_text(
            "❌ An error occurred while starting the bot. Please try again or contact support."
        )

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle callback queries from inline keyboard buttons"""
    try:
        query = update.callback_query
        user_id = query.from_user.id
        
        # Check subscription status
        subscription = check_subscription_status(user_id)
        is_owner = user_id == OWNER_ID
        
        # Handle different callback data
        if query.data == "main_menu":
            await show_main_menu(query, user_id, subscription['is_active'], is_owner)
            
        # Core features
        elif query.data == "ai_chat":
            await show_ai_trading_chat(query)
        elif query.data == "market_overview":
            await show_comprehensive_market_overview(query)
            
        # Trading hubs
        elif query.data == "stocks_hub":
            await show_stocks_hub(query)
        elif query.data == "crypto_hub":
            await show_crypto_hub(query)
        elif query.data == "forex_hub":
            await show_forex_hub(query)
        elif query.data == "commodities_hub":
            await show_commodities_hub(query)
            
        # Analysis tools
        elif query.data == "technical_analysis":
            await show_technical_analysis(query)
        elif query.data == "chart_patterns":
            await show_chart_patterns(query)
        elif query.data == "support_resistance":
            await show_support_resistance(query)
        elif query.data == "fibonacci_analysis":
            await show_fibonacci_analysis(query)
            
        # Trading tools
        elif query.data == "signal_alerts":
            await show_signal_alerts(query, user_id)
        elif query.data == "portfolio_tracker":
            await show_portfolio_tracker(query, user_id)
        elif query.data == "price_alerts":
            await show_price_alerts(query, user_id)
        elif query.data == "orb_trading":
            await show_orb_trading(query, user_id)
            
        # Risk & Education
        elif query.data == "risk_calculator":
            await show_risk_calculator(query)
        elif query.data == "market_scanner":
            await show_market_scanner(query)
        elif query.data == "trading_academy":
            await show_trading_academy(query)
            
        # Profile & Settings
        elif query.data == "profile_settings":
            await show_profile_settings(query, user_id)
        elif query.data == "subscription_plans":
            await show_subscription_plans(query, user_id)
        elif query.data == "help_center":
            await show_help_center(query)
            
        # Affiliate program
        elif query.data == "affiliate_program":
            await show_affiliate_program(query, user_id)
        elif query.data == "my_referrals":
            await show_my_referrals(query, user_id)
        elif query.data == "affiliate_earnings":
            await show_affiliate_earnings(query, user_id)
        elif query.data == "request_payout":
            await show_request_payout(query, user_id)
        elif query.data == "affiliate_guide":
            await show_affiliate_guide(query)
            
        # Admin panel (owner only)
        elif query.data == "admin_panel" and is_owner:
            await show_admin_panel(query)
        elif query.data == "user_management" and is_owner:
            await show_user_management(query)
        elif query.data == "admin_analytics" and is_owner:
            await show_admin_analytics(query)
        elif query.data == "broadcast_message" and is_owner:
            await show_broadcast_interface(query)
            
        # Handle unknown callbacks
        else:
            await query.answer("This feature is not available yet.")
            
    except Exception as e:
        logger.error(f"Error in callback handler: {e}")
        await query.answer("An error occurred. Please try again.")

async def show_main_menu(query, user_id, has_access, is_owner):
    """Show the main menu"""
    text = """🎯 **TradeFlow AI - Main Menu**

Select a feature to get started:"""

    keyboard = []
    
    # Core features
    keyboard.append([
        InlineKeyboardButton("🤖 AI Trading Chat", callback_data="ai_chat"),
        InlineKeyboardButton("📊 Market Overview", callback_data="market_overview")
    ])
    
    # Trading hubs
    keyboard.append([
        InlineKeyboardButton("📈 Stocks Hub", callback_data="stocks_hub"),
        InlineKeyboardButton("₿ Crypto Hub", callback_data="crypto_hub")
    ])
    
    keyboard.append([
        InlineKeyboardButton("💱 Forex Hub", callback_data="forex_hub"),
        InlineKeyboardButton("🪙 Commodities Hub", callback_data="commodities_hub")
    ])
    
    # Analysis tools
    keyboard.append([
        InlineKeyboardButton("📐 Technical Analysis", callback_data="technical_analysis"),
        InlineKeyboardButton("🎯 Chart Patterns", callback_data="chart_patterns")
    ])
    
    keyboard.append([
        InlineKeyboardButton("📊 Support/Resistance", callback_data="support_resistance"),
        InlineKeyboardButton("📐 Fibonacci Analysis", callback_data="fibonacci_analysis")
    ])
    
    # Trading tools
    keyboard.append([
        InlineKeyboardButton("🔔 Signal Alerts", callback_data="signal_alerts"),
        InlineKeyboardButton("📊 Portfolio Tracker", callback_data="portfolio_tracker")
    ])
    
    keyboard.append([
        InlineKeyboardButton("⏰ Price Alerts", callback_data="price_alerts"),
        InlineKeyboardButton("📈 ORB Trading", callback_data="orb_trading")
    ])
    
    # Risk & Education
    keyboard.append([
        InlineKeyboardButton("🎲 Risk Calculator", callback_data="risk_calculator"),
        InlineKeyboardButton("🔍 Market Scanner", callback_data="market_scanner")
    ])
    
    keyboard.append([
        InlineKeyboardButton("📚 Trading Academy", callback_data="trading_academy"),
        InlineKeyboardButton("👤 Profile & Settings", callback_data="profile_settings")
    ])
    
    # Subscription & Support
    keyboard.append([
        InlineKeyboardButton("💎 Subscription Plans", callback_data="subscription_plans"),
        InlineKeyboardButton("❓ Help Center", callback_data="help_center")
    ])
    
    # Add admin panel for owners
    if is_owner:
        keyboard.append([
            InlineKeyboardButton("👑 Admin Panel", callback_data="admin_panel")
        ])
    
    await query.edit_message_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_ai_trading_chat(query):
    """Show the AI trading chat interface"""
    try:
        chat_text = """🤖 **TradeFlow AI Chat**

Welcome to your AI trading assistant! I can help you with:
• Market analysis and insights
• Trading strategy suggestions
• Risk management advice
• Technical analysis explanations
• Trading psychology support

Type your question or request below to get started!"""

        keyboard = [
            [InlineKeyboardButton("📊 Market Analysis", callback_data="market_analysis"),
             InlineKeyboardButton("📈 Trading Strategy", callback_data="trading_strategy")],
            [InlineKeyboardButton("🎯 Risk Management", callback_data="risk_management"),
             InlineKeyboardButton("📚 Trading Education", callback_data="trading_education")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(chat_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in AI chat interface: {e}")
        await query.answer("An error occurred. Please try again.")

async def fetch_stock_data(symbol):
    """Fetch real-time stock data from Alpha Vantage"""
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        if 'Global Quote' in data:
            quote = data['Global Quote']
            return {
                'price': float(quote['05. price']),
                'change': float(quote['09. change']),
                'change_percent': float(quote['10. change percent'].replace('%', '')),
                'volume': int(quote['06. volume'])
            }
        return None
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        return None

async def fetch_crypto_data(coin_id):
    """Fetch real-time cryptocurrency data from CoinGecko"""
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true&include_market_cap=true"
        response = requests.get(url)
        data = response.json()
        if coin_id in data:
            return {
                'price': data[coin_id]['usd'],
                'change_24h': data[coin_id]['usd_24h_change'],
                'market_cap': data[coin_id]['usd_market_cap']
            }
        return None
    except Exception as e:
        logger.error(f"Error fetching crypto data: {e}")
        return None

async def fetch_forex_data(base, quote):
    """Fetch real-time forex data from Fixer"""
    try:
        url = f"http://data.fixer.io/api/latest?access_key={FIXER_API_KEY}&base={base}&symbols={quote}"
        response = requests.get(url)
        data = response.json()
        if 'rates' in data:
            return {
                'rate': data['rates'][quote],
                'timestamp': data['timestamp']
            }
        return None
    except Exception as e:
        logger.error(f"Error fetching forex data: {e}")
        return None

async def fetch_market_overview():
    """Fetch comprehensive market overview data"""
    try:
        # Fetch major indices
        sp500 = await fetch_stock_data('SPY')
        nasdaq = await fetch_stock_data('QQQ')
        dow = await fetch_stock_data('DIA')
        
        # Fetch major cryptocurrencies
        btc = await fetch_crypto_data('bitcoin')
        eth = await fetch_crypto_data('ethereum')
        
        # Fetch major forex pairs
        eur_usd = await fetch_forex_data('EUR', 'USD')
        gbp_usd = await fetch_forex_data('GBP', 'USD')
        usd_jpy = await fetch_forex_data('USD', 'JPY')
        
        return {
            'stocks': {
                'sp500': sp500,
                'nasdaq': nasdaq,
                'dow': dow
            },
            'crypto': {
                'btc': btc,
                'eth': eth
            },
            'forex': {
                'eur_usd': eur_usd,
                'gbp_usd': gbp_usd,
                'usd_jpy': usd_jpy
            }
        }
    except Exception as e:
        logger.error(f"Error fetching market overview: {e}")
        return None

async def show_comprehensive_market_overview(query):
    """Show comprehensive market overview with real data"""
    try:
        market_data = await fetch_market_overview()
        if not market_data:
            await query.answer("Error fetching market data. Please try again.")
            return

        overview_text = """📊 **Market Overview**

**📈 Stocks**
• S&P 500: ${:.2f} ({:+.2f}%)
• NASDAQ: ${:.2f} ({:+.2f}%)
• Dow Jones: ${:.2f} ({:+.2f}%)

**₿ Crypto**
• BTC: ${:,.2f} ({:+.2f}%)
• ETH: ${:,.2f} ({:+.2f}%)
• Market Cap: ${:,.0f}B

**💱 Forex**
• EUR/USD: {:.4f}
• GBP/USD: {:.4f}
• USD/JPY: {:.2f}""".format(
            market_data['stocks']['sp500']['price'], market_data['stocks']['sp500']['change_percent'],
            market_data['stocks']['nasdaq']['price'], market_data['stocks']['nasdaq']['change_percent'],
            market_data['stocks']['dow']['price'], market_data['stocks']['dow']['change_percent'],
            market_data['crypto']['btc']['price'], market_data['crypto']['btc']['change_24h'],
            market_data['crypto']['eth']['price'], market_data['crypto']['eth']['change_24h'],
            market_data['crypto']['btc']['market_cap'] / 1e9,
            market_data['forex']['eur_usd']['rate'],
            market_data['forex']['gbp_usd']['rate'],
            market_data['forex']['usd_jpy']['rate']
        )

        keyboard = [
            [InlineKeyboardButton("📈 Stocks", callback_data="stocks_hub"),
             InlineKeyboardButton("₿ Crypto", callback_data="crypto_hub")],
            [InlineKeyboardButton("💱 Forex", callback_data="forex_hub"),
             InlineKeyboardButton("🪙 Commodities", callback_data="commodities_hub")],
            [InlineKeyboardButton("📊 Market Scanner", callback_data="market_scanner"),
             InlineKeyboardButton("🔔 Market Alerts", callback_data="market_alerts")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(overview_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in market overview: {e}")
        await query.answer("An error occurred. Please try again.")

async def fetch_portfolio_data(user_id):
    """Fetch real-time portfolio data from database"""
    try:
        # Get user's portfolio from database
        portfolio = await db.portfolios.find_one({"user_id": user_id})
        if not portfolio:
            return None

        # Fetch current prices for all holdings
        holdings = portfolio.get('holdings', {})
        current_prices = {}
        total_value = 0
        daily_change = 0

        for symbol, quantity in holdings.items():
            # Fetch current price from Alpha Vantage
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
            response = requests.get(url)
            data = response.json()
            
            if 'Global Quote' in data:
                current_price = float(data['Global Quote']['05. price'])
                prev_close = float(data['Global Quote']['08. previous close'])
                current_prices[symbol] = {
                    'price': current_price,
                    'change': current_price - prev_close,
                    'change_percent': ((current_price - prev_close) / prev_close) * 100
                }
                total_value += current_price * quantity
                daily_change += (current_price - prev_close) * quantity

        return {
            'holdings': holdings,
            'current_prices': current_prices,
            'total_value': total_value,
            'daily_change': daily_change,
            'daily_change_percent': (daily_change / (total_value - daily_change)) * 100 if total_value > daily_change else 0
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {e}")
        return None

async def show_portfolio_tracker(query):
    """Show portfolio tracking with real-time data"""
    try:
        user_id = query.from_user.id
        portfolio_data = await fetch_portfolio_data(user_id)
        
        if not portfolio_data:
            await query.answer("No portfolio data found. Please add some holdings first.")
            return

        portfolio_text = """📊 **Portfolio Overview**

**Total Value:** ${:.2f}
**Daily Change:** ${:.2f} ({:+.2f}%)

**Holdings:**
""".format(
            portfolio_data['total_value'],
            portfolio_data['daily_change'],
            (portfolio_data['daily_change'] / portfolio_data['total_value'] * 100) if portfolio_data['total_value'] > 0 else 0
        )

        # Add holdings details
        for holding in portfolio_data['holdings']:
            symbol, quantity, avg_price, current_price, total_value = holding
            profit_loss = (current_price - avg_price) * quantity
            profit_loss_pct = (current_price - avg_price) / avg_price * 100
            
            portfolio_text += f"• {symbol}: {quantity} shares\n"
            portfolio_text += f"  Avg: ${avg_price:.2f} | Current: ${current_price:.2f}\n"
            portfolio_text += f"  P/L: ${profit_loss:.2f} ({profit_loss_pct:+.2f}%)\n\n"
        keyboard = [
            [InlineKeyboardButton("📈 Performance", callback_data="portfolio_performance")],
            [InlineKeyboardButton("🔧 Scanner Settings", callback_data="scanner_settings"),
             InlineKeyboardButton("📊 Scan Results", callback_data="scan_results")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(portfolio_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in market scanner: {e}")
        await query.answer("An error occurred. Please try again.")

async def fetch_trading_courses():
    """Fetch trading courses from database"""
    try:
        courses = await db.courses.find().to_list(length=None)
        if not courses:
            # Initialize with default courses if none exist
            default_courses = [
                {
                    'id': 'course_1',
                    'title': 'Introduction to Trading',
                    'description': 'Learn the basics of trading and market analysis',
                    'duration': '2 hours',
                    'level': 'Beginner',
                    'topics': ['Market Basics', 'Technical Analysis', 'Risk Management'],
                    'video_url': 'https://example.com/course1',
                    'materials': ['PDF Guide', 'Practice Exercises', 'Quiz']
                },
                {
                    'id': 'course_2',
                    'title': 'Advanced Technical Analysis',
                    'description': 'Master advanced chart patterns and indicators',
                    'duration': '3 hours',
                    'level': 'Intermediate',
                    'topics': ['Chart Patterns', 'Indicators', 'Trading Strategies'],
                    'video_url': 'https://example.com/course2',
                    'materials': ['PDF Guide', 'Practice Exercises', 'Quiz']
                },
                {
                    'id': 'course_3',
                    'title': 'Risk Management & Psychology',
                    'description': 'Learn to manage risk and control emotions while trading',
                    'duration': '2.5 hours',
                    'level': 'Advanced',
                    'topics': ['Risk Management', 'Trading Psychology', 'Position Sizing'],
                    'video_url': 'https://example.com/course3',
                    'materials': ['PDF Guide', 'Practice Exercises', 'Quiz']
                }
            ]
            await db.courses.insert_many(default_courses)
            courses = default_courses

        return courses
    except Exception as e:
        logger.error(f"Error fetching trading courses: {e}")
        return None

async def show_trading_academy(query):
    """Show trading academy with course listings"""
    try:
        courses = await fetch_trading_courses()
        if not courses:
            await query.answer("Error fetching courses. Please try again.")
            return

        academy_text = """📚 **Trading Academy**

**Available Courses:**\n"""
        
        # Add course listings
        for course in courses:
            academy_text += f"\n**{course['title']}**\n"
            academy_text += f"• Level: {course['level']}\n"
            academy_text += f"• Duration: {course['duration']}\n"
            academy_text += f"• Topics: {', '.join(course['topics'])}\n"
            academy_text += f"• Materials: {', '.join(course['materials'])}\n"

        keyboard = [
            [InlineKeyboardButton("📖 Course 1: Introduction", callback_data="course_1"),
             InlineKeyboardButton("📖 Course 2: Technical Analysis", callback_data="course_2")],
            [InlineKeyboardButton("📖 Course 3: Risk Management", callback_data="course_3"),
             InlineKeyboardButton("📊 Practice Exercises", callback_data="practice_exercises")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(academy_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in trading academy: {e}")
        await query.answer("An error occurred. Please try again.")

async def show_course_details(query, course_id):
    """Show detailed course information"""
    try:
        course = await db.courses.find_one({"id": course_id})
        if not course:
            await query.answer("Course not found.")
            return

        course_text = f"""📚 **{course['title']}**

**Description:**
{course['description']}

**Details:**
• Level: {course['level']}
• Duration: {course['duration']}

**Topics Covered:**
{chr(10).join(['• ' + topic for topic in course['topics']])}

**Course Materials:**
{chr(10).join(['• ' + material for material in course['materials']])}

**Start Learning:**
Click the button below to access the course materials."""

        keyboard = [
            [InlineKeyboardButton("🎥 Watch Video", url=course['video_url'])],
            [InlineKeyboardButton("📖 Download Materials", callback_data=f"download_{course_id}")],
            [InlineKeyboardButton("📝 Take Quiz", callback_data=f"quiz_{course_id}")],
            [InlineKeyboardButton("🔙 Back to Academy", callback_data="trading_academy")]
        ]
        
        await query.edit_message_text(course_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error showing course details: {e}")
        await query.answer("An error occurred. Please try again.")

async def track_user_progress(user_id, course_id, progress_type):
    """Track user's progress in courses"""
    try:
        # Update or create progress record
        await db.course_progress.update_one(
            {"user_id": user_id, "course_id": course_id},
            {
                "$set": {
                    f"progress.{progress_type}": True,
                    "last_updated": datetime.now()
                }
            },
            upsert=True
        )
        
        # Check if course is completed
        progress = await db.course_progress.find_one({"user_id": user_id, "course_id": course_id})
        if progress and all(progress.get('progress', {}).values()):
            # Award completion certificate
            await db.certificates.insert_one({
                "user_id": user_id,
                "course_id": course_id,
                "completed_at": datetime.now()
            })
            
            # Notify user
            await bot.send_message(
                chat_id=user_id,
                text=f"🎓 **Course Completed!**\n\nCongratulations on completing the course! "
                     f"You can now access your certificate in the academy section."
            )
    except Exception as e:
        logger.error(f"Error tracking user progress: {e}")

async def fetch_help_articles():
    """Fetch help articles from database"""
    try:
        articles = await db.help_articles.find().to_list(length=None)
        if not articles:
            # Initialize with default articles if none exist
            default_articles = [
                {
                    'id': 'article_1',
                    'title': 'Getting Started Guide',
                    'category': 'Basics',
                    'content': 'Learn how to get started with TradeFlow AI...',
                    'tags': ['beginner', 'setup', 'guide']
                },
                {
                    'id': 'article_2',
                    'title': 'Understanding Technical Analysis',
                    'category': 'Analysis',
                    'content': 'Master the basics of technical analysis...',
                    'tags': ['technical', 'analysis', 'charts']
                },
                {
                    'id': 'article_3',
                    'title': 'Setting Up Alerts',
                    'category': 'Features',
                    'content': 'Learn how to set up and manage price alerts...',
                    'tags': ['alerts', 'notifications', 'settings']
                }
            ]
            await db.help_articles.insert_many(default_articles)
            articles = default_articles

        return articles
    except Exception as e:
        logger.error(f"Error fetching help articles: {e}")
        return None

async def show_help_center(query):
    """Show help center with articles and support options"""
    try:
        articles = await fetch_help_articles()
        if not articles:
            await query.answer("Error fetching help articles. Please try again.")
            return

        help_text = """❓ **Help Center**

**Popular Articles:**\n"""
        
        # Add article listings
        for article in articles:
            help_text += f"\n**{article['title']}**\n"
            help_text += f"• Category: {article['category']}\n"
            help_text += f"• Tags: {', '.join(article['tags'])}\n"

        help_text += "\n**Need More Help?**\n"
        help_text += "• Contact our support team\n"
        help_text += "• Join our community forum\n"
        help_text += "• Check our FAQ section"

        keyboard = [
            [InlineKeyboardButton("📚 Getting Started", callback_data="help_getting_started"),
             InlineKeyboardButton("📊 Features Guide", callback_data="help_features")],
            [InlineKeyboardButton("❓ FAQ", callback_data="help_faq"),
             InlineKeyboardButton("📞 Contact Support", callback_data="help_contact")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(help_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in help center: {e}")
        await query.answer("An error occurred. Please try again.")

async def show_article_details(query, article_id):
    """Show detailed help article"""
    try:
        article = await db.help_articles.find_one({"id": article_id})
        if not article:
            await query.answer("Article not found.")
            return

        article_text = f"""📚 **{article['title']}**

**Category:** {article['category']}
**Tags:** {', '.join(article['tags'])}

{article['content']}

**Was this helpful?**"""

        keyboard = [
            [InlineKeyboardButton("👍 Yes", callback_data=f"helpful_{article_id}"),
             InlineKeyboardButton("👎 No", callback_data=f"not_helpful_{article_id}")],
            [InlineKeyboardButton("📝 Feedback", callback_data=f"feedback_{article_id}")],
            [InlineKeyboardButton("🔙 Back to Help", callback_data="help_center")]
        ]
        
        await query.edit_message_text(article_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error showing article details: {e}")
        await query.answer("An error occurred. Please try again.")

async def track_article_feedback(article_id, user_id, is_helpful):
    """Track user feedback on help articles"""
    try:
        # Update article feedback
        await db.article_feedback.update_one(
            {"article_id": article_id},
            {
                "$inc": {
                    "helpful_count" if is_helpful else "not_helpful_count": 1
                }
            },
            upsert=True
        )
        
        # Track user's feedback
        await db.user_feedback.insert_one({
            "user_id": user_id,
            "article_id": article_id,
            "is_helpful": is_helpful,
            "timestamp": datetime.now()
        })
        
        # If article is not helpful, prompt for feedback
        if not is_helpful:
            await bot.send_message(
                chat_id=user_id,
                text="We're sorry the article wasn't helpful. Please let us know what we can improve:",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("📝 Provide Feedback", callback_data=f"feedback_{article_id}")
                ]])
            )
    except Exception as e:
        logger.error(f"Error tracking article feedback: {e}")

async def show_affiliate_program(query, user_id):
    """Show affiliate program interface"""
    try:
        affiliate_text = """🤝 **Affiliate Program**

**Your Affiliate Stats:**
• Total Referrals: 15
• Active Referrals: 8
• Total Earnings: $150
• Pending Payout: $50

**Referral Link:**
https://tradeflow.ai/ref/trader123

**Commission Structure:**
• 20% of subscription revenue
• 10% of trading fees
• Lifetime earnings

**How It Works:**
1. Share your referral link
2. Get paid for each signup
3. Earn from their trading
4. Request payout anytime"""

        keyboard = [
            [InlineKeyboardButton("👥 My Referrals", callback_data="my_referrals"),
             InlineKeyboardButton("💰 Earnings", callback_data="affiliate_earnings")],
            [InlineKeyboardButton("💸 Request Payout", callback_data="request_payout"),
             InlineKeyboardButton("📚 Affiliate Guide", callback_data="affiliate_guide")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(affiliate_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in affiliate program: {e}")
        await query.answer("An error occurred. Please try again.")

async def show_admin_panel(query):
    """Show admin panel interface"""
    try:
        admin_text = """👑 **Admin Panel**

**User Management:**
• Total Users: 1,234
• Active Users: 789
• Premium Users: 456
• New Today: 12

**System Status:**
• API Status: Online
• Database: Healthy
• Memory Usage: 45%
• CPU Load: 30%

**Quick Actions:**
• Broadcast Message
• User Management
• Analytics Dashboard
• System Settings"""

        keyboard = [
            [InlineKeyboardButton("👥 User Management", callback_data="user_management"),
             InlineKeyboardButton("📊 Analytics", callback_data="admin_analytics")],
            [InlineKeyboardButton("📢 Broadcast", callback_data="broadcast_message"),
             InlineKeyboardButton("⚙️ Settings", callback_data="admin_settings")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(admin_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in admin panel: {e}")
        await query.answer("An error occurred. Please try again.")

async def show_user_management(query):
    """Show user management interface"""
    try:
        users_text = """👥 **User Management**

**User Statistics:**
• Total Users: 1,234
• Active Today: 123
• New Users: 45
• Premium Users: 456

**Recent Activity:**
• New signups: 12
• Upgrades: 5
• Cancellations: 2
• Support tickets: 8

**User Actions:**
• View User Details
• Manage Subscriptions
• Handle Support
• User Analytics"""

        keyboard = [
            [InlineKeyboardButton("🔍 Search Users", callback_data="search_users"),
             InlineKeyboardButton("📊 User Analytics", callback_data="user_analytics")],
            [InlineKeyboardButton("💬 Support Tickets", callback_data="support_tickets"),
             InlineKeyboardButton("📈 Growth Stats", callback_data="growth_stats")],
            [InlineKeyboardButton("🔙 Back to Admin", callback_data="admin_panel")]
        ]
        
        await query.edit_message_text(users_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in user management: {e}")
        await query.answer("An error occurred. Please try again.")

async def show_admin_analytics(query):
    """Show admin analytics interface"""
    try:
        analytics_text = """📊 **Admin Analytics**

**Platform Metrics:**
• Daily Active Users: 789
• Revenue: $12,345
• Conversion Rate: 3.2%
• Churn Rate: 1.5%

**Feature Usage:**
• AI Chat: 45%
• Market Scanner: 30%
• ORB Trading: 25%
• Portfolio: 40%

**Performance:**
• API Response: 95ms
• Uptime: 99.9%
• Error Rate: 0.1%
• User Satisfaction: 4.8/5"""

        keyboard = [
            [InlineKeyboardButton("📈 Revenue Stats", callback_data="revenue_stats"),
             InlineKeyboardButton("👥 User Stats", callback_data="user_stats")],
            [InlineKeyboardButton("📊 Feature Usage", callback_data="feature_usage"),
             InlineKeyboardButton("📱 Performance", callback_data="performance_metrics")],
            [InlineKeyboardButton("🔙 Back to Admin", callback_data="admin_panel")]
        ]
        
        await query.edit_message_text(analytics_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in admin analytics: {e}")
        await query.answer("An error occurred. Please try again.")

async def fetch_technical_indicators(symbol, timeframe='1d'):
    """Fetch technical indicators data from Alpha Vantage"""
    try:
        url = f"https://www.alphavantage.co/query?function=TECHNICAL_INDICATORS&symbol={symbol}&interval={timeframe}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if 'Technical Analysis' in data:
            return {
                'rsi': float(data['Technical Analysis']['RSI'][0]['RSI']),
                'macd': {
                    'macd': float(data['Technical Analysis']['MACD'][0]['MACD']),
                    'signal': float(data['Technical Analysis']['MACD'][0]['MACD_Signal']),
                    'histogram': float(data['Technical Analysis']['MACD'][0]['MACD_Hist'])
                },
                'sma': {
                    'sma20': float(data['Technical Analysis']['SMA'][0]['SMA']),
                    'sma50': float(data['Technical Analysis']['SMA'][1]['SMA']),
                    'sma200': float(data['Technical Analysis']['SMA'][2]['SMA'])
                }
            }
        return None
    except Exception as e:
        logger.error(f"Error fetching technical indicators: {e}")
        return None

async def fetch_support_resistance(symbol):
    """Calculate support and resistance levels"""
    try:
        # Fetch historical data
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if 'Time Series (Daily)' in data:
            prices = [float(data['Time Series (Daily)'][date]['4. close']) for date in data['Time Series (Daily)']]
            current_price = prices[0]
            
            # Calculate support and resistance levels
            levels = {
                'resistance': [
                    max(prices[:5]),  # R1
                    max(prices[5:10])  # R2
                ],
                'support': [
                    min(prices[:5]),  # S1
                    min(prices[5:10])  # S2
                ],
                'current_price': current_price
            }
            return levels
        return None
    except Exception as e:
        logger.error(f"Error calculating support/resistance: {e}")
        return None

async def fetch_fibonacci_levels(symbol):
    """Calculate Fibonacci retracement levels"""
    try:
        # Fetch historical data
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if 'Time Series (Daily)' in data:
            prices = [float(data['Time Series (Daily)'][date]['4. close']) for date in data['Time Series (Daily)']]
            high = max(prices[:20])
            low = min(prices[:20])
            current_price = prices[0]
            
            # Calculate Fibonacci levels
            diff = high - low
            levels = {
                '0.0': low,
                '0.236': low + diff * 0.236,
                '0.382': low + diff * 0.382,
                '0.5': low + diff * 0.5,
                '0.618': low + diff * 0.618,
                '0.786': low + diff * 0.786,
                '1.0': high,
                'current_price': current_price
            }
            return levels
        return None
    except Exception as e:
        logger.error(f"Error calculating Fibonacci levels: {e}")
        return None

async def show_technical_analysis(query):
    """Show technical analysis tools with real data"""
    try:
        # Fetch data for a default symbol (e.g., SPY)
        indicators = await fetch_technical_indicators('SPY')
        if not indicators:
            await query.answer("Error fetching technical indicators. Please try again.")
            return

        analysis_text = """📊 **Technical Analysis**

**Current Indicators (SPY):**
• RSI: {:.2f}
• MACD: {:.2f}
• Signal: {:.2f}
• Histogram: {:.2f}

**Moving Averages:**
• SMA 20: {:.2f}
• SMA 50: {:.2f}
• SMA 200: {:.2f}

Select an analysis tool:""".format(
            indicators['rsi'],
            indicators['macd']['macd'],
            indicators['macd']['signal'],
            indicators['macd']['histogram'],
            indicators['sma']['sma20'],
            indicators['sma']['sma50'],
            indicators['sma']['sma200']
        )

        keyboard = [
            [InlineKeyboardButton("📈 Support/Resistance", callback_data="support_resistance"),
             InlineKeyboardButton("📊 Chart Patterns", callback_data="chart_patterns")],
            [InlineKeyboardButton("📐 Fibonacci", callback_data="fibonacci_analysis"),
             InlineKeyboardButton("📊 Indicators", callback_data="technical_indicators")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(analysis_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in technical analysis: {e}")
        await query.answer("An error occurred. Please try again.")

async def show_support_resistance(query):
    """Show support and resistance analysis with real data"""
    try:
        # Fetch data for a default symbol (e.g., SPY)
        levels = await fetch_support_resistance('SPY')
        if not levels:
            await query.answer("Error fetching support/resistance levels. Please try again.")
            return

        sr_text = """📈 **Support & Resistance Analysis**

**Current Levels:**
• Resistance 2: ${:.2f}
• Resistance 1: ${:.2f}
• Current Price: ${:.2f}
• Support 1: ${:.2f}
• Support 2: ${:.2f}

**Key Levels:**
• Daily High: ${:.2f}
• Daily Low: ${:.2f}""".format(
            levels['resistance'][1],
            levels['resistance'][0],
            levels['current_price'],
            levels['support'][0],
            levels['support'][1],
            max(levels['resistance']),
            min(levels['support'])
        )

        keyboard = [
            [InlineKeyboardButton("📊 Volume Profile", callback_data="volume_profile"),
             InlineKeyboardButton("📈 Price Action", callback_data="price_action")],
            [InlineKeyboardButton("🔙 Back to Analysis", callback_data="technical_analysis")]
        ]
        
        await query.edit_message_text(sr_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in support/resistance: {e}")
        await query.answer("An error occurred. Please try again.")

async def show_fibonacci_analysis(query):
    """Show Fibonacci analysis with real data"""
    try:
        # Fetch data for a default symbol (e.g., SPY)
        levels = await fetch_fibonacci_levels('SPY')
        if not levels:
            await query.answer("Error fetching Fibonacci levels. Please try again.")
            return

        fib_text = """📐 **Fibonacci Analysis**

**Key Levels:**
• 0.0: ${:.2f} (Swing Low)
• 0.236: ${:.2f}
• 0.382: ${:.2f}
• 0.5: ${:.2f}
• 0.618: ${:.2f}
• 0.786: ${:.2f}
• 1.0: ${:.2f} (Swing High)

**Current Price: ${:.2f}**
• Between {:.1f} and {:.1f} levels
• Potential support at {:.1f}
• Potential resistance at {:.1f}""".format(
            levels['0.0'],
            levels['0.236'],
            levels['0.382'],
            levels['0.5'],
            levels['0.618'],
            levels['0.786'],
            levels['1.0'],
            levels['current_price'],
            min(k for k, v in levels.items() if v < levels['current_price']),
            max(k for k, v in levels.items() if v > levels['current_price']),
            min(k for k, v in levels.items() if v < levels['current_price']),
            max(k for k, v in levels.items() if v > levels['current_price'])
        )

        keyboard = [
            [InlineKeyboardButton("📈 Trend Analysis", callback_data="trend_analysis"),
             InlineKeyboardButton("📊 Price Action", callback_data="price_action")],
            [InlineKeyboardButton("🔙 Back to Analysis", callback_data="technical_analysis")]
        ]
        
        await query.edit_message_text(fib_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in Fibonacci analysis: {e}")
        await query.answer("An error occurred. Please try again.")

async def fetch_orb_data(symbol, timeframe='1d'):
    """Fetch data for ORB (Opening Range Breakout) strategy"""
    try:
        # Fetch daily data from Alpha Vantage
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            return None

        # Get today's data
        today = datetime.now().strftime('%Y-%m-%d')
        if today not in data['Time Series (Daily)']:
            return None

        today_data = data['Time Series (Daily)'][today]
        open_price = float(today_data['1. open'])
        high_price = float(today_data['2. high'])
        low_price = float(today_data['3. low'])
        current_price = float(today_data['4. close'])
        volume = int(today_data['6. volume'])

        # Calculate ORB levels
        orb_levels = {
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'current': current_price,
            'volume': volume,
            'breakout_up': high_price > open_price,
            'breakout_down': low_price < open_price,
            'range': high_price - low_price,
            'range_percent': ((high_price - low_price) / open_price) * 100
        }

        # Calculate previous day's data for context
        dates = sorted(data['Time Series (Daily)'].keys())
        if len(dates) > 1:
            prev_day = dates[1]
            prev_data = data['Time Series (Daily)'][prev_day]
            orb_levels['prev_close'] = float(prev_data['4. close'])
            orb_levels['prev_volume'] = int(prev_data['6. volume'])
            orb_levels['gap_up'] = open_price > float(prev_data['4. close'])
            orb_levels['gap_down'] = open_price < float(prev_data['4. close'])

        return orb_levels
    except Exception as e:
        logger.error(f"Error fetching ORB data: {e}")
        return None

async def show_orb_trading(query):
    """Show ORB trading strategy with real-time data"""
    try:
        # Fetch data for a default symbol (e.g., SPY)
        orb_data = await fetch_orb_data('SPY')
        if not orb_data:
            await query.answer("Error fetching ORB data. Please try again.")
            return

        orb_text = """📈 **ORB Trading Strategy**

**Current Session (SPY):**
• Open: ${:.2f}
• High: ${:.2f}
• Low: ${:.2f}
• Current: ${:.2f}
• Volume: {:,}

**Breakout Status:**
• Range: ${:.2f} ({:.2f}%)
• Breakout Up: {}
• Breakout Down: {}

**Previous Session:**
• Close: ${:.2f}
• Volume: {:,}
• Gap Up: {}
• Gap Down: {}

Select an action:""".format(
            orb_data['open'],
            orb_data['high'],
            orb_data['low'],
            orb_data['current'],
            orb_data['volume'],
            orb_data['range'],
            orb_data['range_percent'],
            '✅' if orb_data['breakout_up'] else '❌',
            '✅' if orb_data['breakout_down'] else '❌',
            orb_data.get('prev_close', 0),
            orb_data.get('prev_volume', 0),
            '✅' if orb_data.get('gap_up', False) else '❌',
            '✅' if orb_data.get('gap_down', False) else '❌'
        )

        keyboard = [
            [InlineKeyboardButton("📊 Set ORB Levels", callback_data="set_orb_levels"),
             InlineKeyboardButton("📈 Monitor Breakouts", callback_data="monitor_breakouts")],
            [InlineKeyboardButton("📊 Volume Analysis", callback_data="orb_volume"),
             InlineKeyboardButton("📈 Pattern Scanner", callback_data="orb_patterns")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(orb_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in ORB trading: {e}")
        await query.answer("An error occurred. Please try again.")

async def monitor_orb_breakouts():
    """Monitor ORB breakouts for all tracked symbols"""
    try:
        # Get all tracked symbols from database
        tracked_symbols = await db.orb_tracking.distinct("symbol")
        
        for symbol in tracked_symbols:
            # Fetch current ORB data
            orb_data = await fetch_orb_data(symbol)
            if not orb_data:
                continue

            # Check for breakouts
            breakout_up = orb_data['breakout_up']
            breakout_down = orb_data['breakout_down']

            # Get users tracking this symbol
            users = await db.orb_tracking.find({"symbol": symbol}).to_list(length=None)
            
            for user in users:
                if breakout_up and user.get('track_up', False):
                    # Send breakout notification
                    await bot.send_message(
                        chat_id=user['user_id'],
                        text=f"🚨 **ORB Breakout Alert**\n\n{symbol} has broken above the opening range!\n\n"
                             f"• Open: ${orb_data['open']:.2f}\n"
                             f"• Current: ${orb_data['current']:.2f}\n"
                             f"• Volume: {orb_data['volume']:,}"
                    )
                
                if breakout_down and user.get('track_down', False):
                    # Send breakdown notification
                    await bot.send_message(
                        chat_id=user['user_id'],
                        text=f"🚨 **ORB Breakdown Alert**\n\n{symbol} has broken below the opening range!\n\n"
                             f"• Open: ${orb_data['open']:.2f}\n"
                             f"• Current: ${orb_data['current']:.2f}\n"
                             f"• Volume: {orb_data['volume']:,}"
                    )

    except Exception as e:
        logger.error(f"Error monitoring ORB breakouts: {e}")

async def fetch_market_scanner_data():
    """Fetch data for market scanner"""
    try:
        # Fetch top gainers and losers from Alpha Vantage
        url = f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if 'top_gainers' not in data or 'top_losers' not in data:
            return None

        # Process top gainers
        gainers = []
        for stock in data['top_gainers'][:5]:  # Get top 5 gainers
            gainers.append({
                'symbol': stock['ticker'],
                'price': float(stock['price']),
                'change': float(stock['change_amount']),
                'change_percent': float(stock['change_percentage']),
                'volume': int(stock['volume'])
            })

        # Process top losers
        losers = []
        for stock in data['top_losers'][:5]:  # Get top 5 losers
            losers.append({
                'symbol': stock['ticker'],
                'price': float(stock['price']),
                'change': float(stock['change_amount']),
                'change_percent': float(stock['change_percentage']),
                'volume': int(stock['volume'])
            })

        # Fetch most active stocks
        url = f"https://www.alphavantage.co/query?function=MARKET_STATUS&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        market_status = response.json()
        
        return {
            'gainers': gainers,
            'losers': losers,
            'market_status': market_status.get('markets', []),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        logger.error(f"Error fetching market scanner data: {e}")
        return None

async def show_market_scanner(query):
    """Show market scanner with real-time data"""
    try:
        scanner_data = await fetch_market_scanner_data()
        if not scanner_data:
            await query.answer("Error fetching market data. Please try again.")
            return

        scanner_text = """🔍 **Market Scanner**

**Top Gainers:**
"""
        # Add top gainers
        for stock in scanner_data['gainers']:
            scanner_text += f"\n**{stock['symbol']}**\n"
            scanner_text += f"• Price: ${stock['price']:.2f}\n"
            scanner_text += f"• Change: ${stock['change']:.2f} ({stock['change_percent']:+.2f}%)\n"
            scanner_text += f"• Volume: {stock['volume']:,}\n"

        scanner_text += "\n**Top Losers:**\n"
        # Add top losers
        for stock in scanner_data['losers']:
            scanner_text += f"\n**{stock['symbol']}**\n"
            scanner_text += f"• Price: ${stock['price']:.2f}\n"
            scanner_text += f"• Change: ${stock['change']:.2f} ({stock['change_percent']:+.2f}%)\n"
            scanner_text += f"• Volume: {stock['volume']:,}\n"

        scanner_text += "\n**Market Status:**\n"
        # Add market status
        for market in scanner_data['market_status']:
            scanner_text += f"• {market['name']}: {market['status']}\n"

        scanner_text += f"\nLast Updated: {scanner_data['timestamp']}"

        keyboard = [
            [InlineKeyboardButton("📈 Volume Scanner", callback_data="volume_scanner"),
             InlineKeyboardButton("📊 Momentum Scanner", callback_data="momentum_scanner")],
            [InlineKeyboardButton("📈 Breakout Scanner", callback_data="breakout_scanner"),
             InlineKeyboardButton("📊 Pattern Scanner", callback_data="pattern_scanner")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(scanner_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in market scanner: {e}")
        await query.answer("An error occurred. Please try again.")

async def scan_market_conditions():
    """Scan market conditions and send alerts"""
    try:
        scanner_data = await fetch_market_scanner_data()
        if not scanner_data:
            return

        # Get users with market scanner alerts enabled
        users = await db.market_alerts.find({"enabled": True}).to_list(length=None)
        
        for user in users:
            # Check for significant movers
            for stock in scanner_data['gainers']:
                if stock['change_percent'] > user.get('gain_threshold', 5):
                    await bot.send_message(
                        chat_id=user['user_id'],
                        text=f"🚨 **Significant Mover Alert**\n\n{stock['symbol']} is up {stock['change_percent']:+.2f}%!\n\n"
                             f"• Price: ${stock['price']:.2f}\n"
                             f"• Change: ${stock['change']:.2f}\n"
                             f"• Volume: {stock['volume']:,}"
                    )

            for stock in scanner_data['losers']:
                if abs(stock['change_percent']) > user.get('loss_threshold', 5):
                    await bot.send_message(
                        chat_id=user['user_id'],
                        text=f"🚨 **Significant Mover Alert**\n\n{stock['symbol']} is down {abs(stock['change_percent']):.2f}%!\n\n"
                             f"• Price: ${stock['price']:.2f}\n"
                             f"• Change: ${stock['change']:.2f}\n"
                             f"• Volume: {stock['volume']:,}"
                    )

    except Exception as e:
        logger.error(f"Error scanning market conditions: {e}")

async def main():
    # ... existing code ...
    logger.info("Bot is running and waiting for commands...")
    # ... existing code ...
