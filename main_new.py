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
from dotenv import load_dotenv

# Import our modules
from config import (
    TELEGRAM_BOT_TOKEN,
    ALPHA_VANTAGE_API_KEY,
    COINGECKO_API_KEY,
    FIXER_API_KEY,
    GOOGLE_AI_API_KEY,
    NEWS_API_KEY,
    MONGODB_URI,
    DB_NAME,
    BOT_OWNER_ID,
    DEFAULT_LANGUAGE,
    SUPPORT_CHAT_ID,
    SUBSCRIPTION_PLANS,
    DEFAULT_TIMEFRAME,
    DEFAULT_SYMBOL,
    MIN_ALERT_PRICE,
    MAX_ALERT_PRICE,
    MIN_POSITION_SIZE,
    MAX_POSITION_SIZE,
    AFFILIATE_COMMISSION_RATE,
    MIN_PAYOUT_AMOUNT,
    REFERRAL_BONUS,
    AI_MODEL,
    AI_TEMPERATURE,
    AI_MAX_TOKENS,
    AI_CHAT_HISTORY_LIMIT,
    MARKET_DATA_REFRESH_INTERVAL,
    PRICE_ALERT_CHECK_INTERVAL,
    TECHNICAL_INDICATORS,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
    CACHE_TTL,
    CACHE_MAX_SIZE,
    API_RATE_LIMIT,
    MAX_LOGIN_ATTEMPTS,
    SESSION_TIMEOUT,
    FEATURES,
    NOTIFICATION_CHANNELS,
    DEFAULT_TIMEZONE,
    MARKET_HOURS,
    AI_CHAT_SETTINGS
)
from models import (
    User, Portfolio, Alert, Subscription, Affiliate,
    Course, HelpArticle, MarketData, TechnicalIndicator,
    ORBLevels, ChatHistory, UserFeedback, PayoutRequest,
    MarketScanner, Notification, Session, ErrorLog, APIRateLimit
)
from database import Database
from api_integrations import MarketDataAPI
from utils import (
    generate_token, verify_token, generate_referral_code,
    hash_password, verify_password, format_currency,
    format_percentage, format_timestamp, format_market_data,
    format_alert, format_error, format_success, format_warning,
    format_info, calculate_profit_loss, calculate_position_size,
    calculate_risk_reward, calculate_moving_average, calculate_rsi,
    calculate_bollinger_bands, calculate_macd, calculate_fibonacci_levels,
    calculate_support_resistance
)
from ai_chat import AIChatHandler
from market_analysis import MarketAnalysis

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global variable to track if bot is running
BOT_RUNNING = False
application = None
INSTANCE_LOCK_FILE = "bot_instance.lock"

# Initialize our modules
db = Database()
market_api = MarketDataAPI()
ai_chat = AIChatHandler(db)
market_analysis = MarketAnalysis(db, market_api)

# Initialize bot
application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

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

async def init_database():
    """Initialize database with required tables"""
    try:
        # Connect to database
        await db.connect()
        
        # Create required collections
        await db.create_collections()
        
        logger.info("✅ Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

async def get_or_create_user(telegram_id: int, username: str = None, first_name: str = None, last_name: str = None, referral_code: str = None):
    """Get or create user in database"""
    try:
        # Check if user exists
        user = await db.get_user(telegram_id)
        if user:
            return user
        
        # Create new user
        user_data = {
            'telegram_id': telegram_id,
            'username': username,
            'first_name': first_name,
            'last_name': last_name,
            'referral_code': referral_code or generate_referral_code(telegram_id),
            'created_at': datetime.now(),
            'last_active': datetime.now(),
            'subscription': {
                'plan': 'free',
                'start_date': datetime.now(),
                'end_date': datetime.now() + timedelta(days=7),
                'status': 'active'
            }
        }
        
        return await db.create_user(user_data)
    except Exception as e:
        logger.error(f"Error in get_or_create_user: {e}")
        return None

async def check_subscription_status(user_id: int):
    """Check user's subscription status"""
    try:
        user = await db.get_user(user_id)
        if not user:
            return {
                'status': 'inactive',
                'is_active': False,
                'days_remaining': 0
            }
        
        subscription = user.get('subscription', {})
        if not subscription:
            return {
                'status': 'inactive',
                'is_active': False,
                'days_remaining': 0
            }
        
        end_date = subscription.get('end_date')
        if not end_date:
            return {
                'status': 'inactive',
                'is_active': False,
                'days_remaining': 0
            }
        
        now = datetime.now()
        days_remaining = (end_date - now).days
        
        return {
            'status': subscription.get('plan', 'free'),
            'is_active': days_remaining > 0,
            'days_remaining': max(0, days_remaining)
        }
    except Exception as e:
        logger.error(f"Error checking subscription status: {e}")
        return {
            'status': 'inactive',
            'is_active': False,
            'days_remaining': 0
        }

async def chat_with_ai(message: str, user_id: int = None, telegram_id: int = None) -> str:
    """Chat with AI using the new AI chat handler"""
    try:
        # Get chat history
        chat_history = await db.get_chat_history(user_id or telegram_id)
        
        # Process message with AI chat handler
        response = await ai_chat.process_message(
            message=message,
            user_id=user_id or telegram_id,
            chat_history=chat_history
        )
        
        # Save conversation
        await db.save_conversation(
            user_id=user_id,
            telegram_id=telegram_id,
            user_message=message,
            ai_response=response,
            context_used=ai_chat.last_context,
            provider_used=ai_chat.last_provider,
            session_id=ai_chat.session_id
        )
        
        return response
    except Exception as e:
        logger.error(f"Error in chat_with_ai: {e}")
        return ERROR_MESSAGES['ai_error']

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
        [InlineKeyboardButton("💸 Affiliate Program", callback_data="affiliate_program")]
    ]
    if is_owner:
        keyboard.append([InlineKeyboardButton("👑 Admin Panel", callback_data="admin_panel")])
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command"""
    try:
        # Get user info
        user = update.effective_user
        if not user:
            await update.message.reply_text("❌ Error: Could not identify user.")
            return
        
        # Get or create user in database
        user_data = await get_or_create_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name
        )
        
        # Check subscription status
        subscription = await check_subscription_status(user.id)
        
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
            is_owner=user.id == BOT_OWNER_ID
        )
        
        # Send welcome message with menu
        await update.message.reply_text(
            welcome_text,
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
        
        # Log user start
        logger.info(f"✅ User {user.id} started the bot")
        
    except Exception as e:
        logger.error(f"Error in start command: {e}")
        await update.message.reply_text(
            "❌ An error occurred while starting the bot. Please try again or contact support."
        )

async def show_ai_trading_chat(query):
    """Show AI trading chat interface"""
    try:
        # Get user ID
        user_id = query.from_user.id
        
        # Get chat history
        chat_history = await db.get_chat_history(user_id)
        
        # Initialize AI chat if needed
        if not ai_chat.is_initialized:
            await ai_chat.initialize_ai()
        
        # Create keyboard
        keyboard = [
            [InlineKeyboardButton("📊 Market Analysis", callback_data="ai_market_analysis")],
            [InlineKeyboardButton("📈 Technical Analysis", callback_data="ai_technical_analysis")],
            [InlineKeyboardButton("📰 News Impact", callback_data="ai_news_impact")],
            [InlineKeyboardButton("🎯 Trading Strategy", callback_data="ai_trading_strategy")],
            [InlineKeyboardButton("📚 Education", callback_data="ai_education")],
            [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="main_menu")]
        ]
        
        # Send message
        await query.edit_message_text(
            text="🤖 *AI Trading Chat*\n\n"
                 "I'm your AI trading assistant. How can I help you today?\n\n"
                 "You can ask me about:\n"
                 "• Market analysis and trends\n"
                 "• Technical indicators and patterns\n"
                 "• News impact on markets\n"
                 "• Trading strategies and setups\n"
                 "• Trading education and concepts\n\n"
                 "Or simply chat with me about any trading topic!",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Error in show_ai_trading_chat: {e}")
        await query.answer("❌ An error occurred. Please try again.")

async def show_comprehensive_market_overview(query):
    """Show comprehensive market overview"""
    try:
        # Get market overview
        overview = await market_analysis.get_market_overview()
        if not overview:
            await query.answer("❌ Error fetching market data. Please try again.")
            return
        
        # Format message
        message = "📊 *Market Overview*\n\n"
        
        # Stocks
        if overview.get('stocks'):
            message += "*Stocks:*\n"
            for symbol, data in overview['stocks'].items():
                message += f"• {symbol}: {format_currency(data['price'])} ({format_percentage(data['change'])})\n"
            message += "\n"
        
        # Crypto
        if overview.get('crypto'):
            message += "*Cryptocurrencies:*\n"
            for symbol, data in overview['crypto'].items():
                message += f"• {symbol}: {format_currency(data['price'])} ({format_percentage(data['change'])})\n"
            message += "\n"
        
        # Forex
        if overview.get('forex'):
            message += "*Forex:*\n"
            for pair, data in overview['forex'].items():
                message += f"• {pair}: {format_currency(data['rate'])} ({format_percentage(data['change'])})\n"
            message += "\n"
        
        # News
        if overview.get('news'):
            message += "*Latest News:*\n"
            for news in overview['news'][:3]:
                message += f"• {news['title']}\n"
        
        # Create keyboard
        keyboard = [
            [InlineKeyboardButton("📈 Stocks", callback_data="stock_market"),
             InlineKeyboardButton("₿ Crypto", callback_data="crypto_market")],
            [InlineKeyboardButton("💱 Forex", callback_data="forex_market"),
             InlineKeyboardButton("📰 News", callback_data="market_news")],
            [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="main_menu")]
        ]
        
        # Send message
        await query.edit_message_text(
            text=message,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Error in show_comprehensive_market_overview: {e}")
        await query.answer("❌ An error occurred. Please try again.")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle callback queries"""
    try:
        query = update.callback_query
        await query.answer()
        
        # Get user data
        user_id = query.from_user.id
        user_data = await get_or_create_user(user_id)
        
        # Check subscription status
        subscription = await check_subscription_status(user_id)
        has_access = subscription['is_active']
        is_owner = user_id == BOT_OWNER_ID
        
        # Handle different callback data
        if query.data == "main_menu":
            await show_main_menu(query, user_id, has_access, is_owner)
        elif query.data == "ai_trading_chat":
            await show_ai_trading_chat(query)
        elif query.data == "market_overview":
            await show_comprehensive_market_overview(query)
        elif query.data == "trading_hub":
            await show_trading_hub(query)
        elif query.data == "analysis_hub":
            await show_analysis_hub(query)
        elif query.data == "portfolio_tracker":
            await show_portfolio_tracker(query, user_id)
        elif query.data == "orb_trading":
            await show_orb_trading(query)
        elif query.data == "price_alerts":
            await show_price_alerts(query, user_id)
        elif query.data == "automation_hub":
            await show_automation_hub(query)
        elif query.data == "trading_academy":
            await show_trading_academy(query)
        elif query.data == "help_center":
            await show_help_center(query)
        elif query.data == "profile_settings":
            await show_profile_settings(query, user_id)
        elif query.data == "subscription_plans":
            await show_subscription_plans(query)
        elif query.data == "affiliate_program":
            await show_affiliate_program(query, user_id)
        elif query.data == "admin_panel" and is_owner:
            await show_admin_panel(query)
        else:
            await query.answer("❌ Invalid option selected.")
            
    except Exception as e:
        logger.error(f"Error in handle_callback: {e}")
        await query.answer("❌ An error occurred. Please try again.")

async def main():
    """Main function to start the bot"""
    try:
        # Check for single instance
        if not check_single_instance():
            return
        
        # Initialize database
        if not await init_database():
            return
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CallbackQueryHandler(handle_callback))
        
        # Start the bot
        global BOT_RUNNING
        BOT_RUNNING = True
        
        logger.info("🚀 Starting TradeFlow AI Bot...")
        await application.run_polling()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Clean up
        if os.path.exists(INSTANCE_LOCK_FILE):
            try:
                os.remove(INSTANCE_LOCK_FILE)
            except:
                pass

if __name__ == "__main__":
    # Initialize database and other async setup
    asyncio.run(init_database())
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(handle_callback))
    # Start the bot (this manages its own event loop)
    application.run_polling() 