from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging
from config import MONGODB_URI, DB_NAME
from models import (
    User, Portfolio, Alert, Subscription,
    Affiliate, Course, HelpArticle, MarketData,
    TechnicalIndicator, ORBLevels, ChatHistory,
    UserFeedback, PayoutRequest, MarketScanner,
    Notification, Session, ErrorLog, APIRateLimit
)

logger = logging.getLogger(__name__)

class Database:
    """Handles all database operations"""
    
    def __init__(self):
        self.client = None
        self.db = None

    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(MONGODB_URI)
            self.db = self.client[DB_NAME]
            logger.info("Connected to MongoDB")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise

    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")

    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        try:
            user_data = await self.db.users.find_one({"user_id": user_id})
            return User(**user_data) if user_data else None
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None

    async def create_user(self, user_data: Dict) -> Optional[User]:
        """Create new user"""
        try:
            user = User(**user_data)
            await self.db.users.insert_one(user.dict())
            return user
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None

    async def update_user(self, user_id: int, update_data: Dict) -> bool:
        """Update user data"""
        try:
            result = await self.db.users.update_one(
                {"user_id": user_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return False

    async def get_portfolio(self, user_id: int) -> List[Portfolio]:
        """Get user's portfolio"""
        try:
            cursor = self.db.portfolios.find({"user_id": user_id})
            portfolios = await cursor.to_list(length=None)
            return [Portfolio(**p) for p in portfolios]
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return []

    async def add_portfolio_item(self, portfolio_data: Dict) -> Optional[Portfolio]:
        """Add item to portfolio"""
        try:
            portfolio = Portfolio(**portfolio_data)
            await self.db.portfolios.insert_one(portfolio.dict())
            return portfolio
        except Exception as e:
            logger.error(f"Error adding portfolio item: {e}")
            return None

    async def update_portfolio_item(self, item_id: str, update_data: Dict) -> bool:
        """Update portfolio item"""
        try:
            result = await self.db.portfolios.update_one(
                {"_id": item_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating portfolio item: {e}")
            return False

    async def get_alerts(self, user_id: int) -> List[Alert]:
        """Get user's alerts"""
        try:
            cursor = self.db.alerts.find({"user_id": user_id})
            alerts = await cursor.to_list(length=None)
            return [Alert(**a) for a in alerts]
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []

    async def create_alert(self, alert_data: Dict) -> Optional[Alert]:
        """Create new alert"""
        try:
            alert = Alert(**alert_data)
            await self.db.alerts.insert_one(alert.dict())
            return alert
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return None

    async def delete_alert(self, alert_id: str) -> bool:
        """Delete alert"""
        try:
            result = await self.db.alerts.delete_one({"_id": alert_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting alert: {e}")
            return False

    async def get_subscription(self, user_id: int) -> Optional[Subscription]:
        """Get user's subscription"""
        try:
            sub_data = await self.db.subscriptions.find_one({"user_id": user_id})
            return Subscription(**sub_data) if sub_data else None
        except Exception as e:
            logger.error(f"Error getting subscription: {e}")
            return None

    async def update_subscription(self, user_id: int, update_data: Dict) -> bool:
        """Update subscription"""
        try:
            result = await self.db.subscriptions.update_one(
                {"user_id": user_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating subscription: {e}")
            return False

    async def get_affiliate_data(self, user_id: int) -> Optional[Affiliate]:
        """Get affiliate data"""
        try:
            aff_data = await self.db.affiliates.find_one({"user_id": user_id})
            return Affiliate(**aff_data) if aff_data else None
        except Exception as e:
            logger.error(f"Error getting affiliate data: {e}")
            return None

    async def update_affiliate_data(self, user_id: int, update_data: Dict) -> bool:
        """Update affiliate data"""
        try:
            result = await self.db.affiliates.update_one(
                {"user_id": user_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating affiliate data: {e}")
            return False

    async def get_courses(self) -> List[Course]:
        """Get all courses"""
        try:
            cursor = self.db.courses.find()
            courses = await cursor.to_list(length=None)
            return [Course(**c) for c in courses]
        except Exception as e:
            logger.error(f"Error getting courses: {e}")
            return []

    async def get_help_articles(self, category: Optional[str] = None) -> List[HelpArticle]:
        """Get help articles"""
        try:
            query = {"category": category} if category else {}
            cursor = self.db.help_articles.find(query)
            articles = await cursor.to_list(length=None)
            return [HelpArticle(**a) for a in articles]
        except Exception as e:
            logger.error(f"Error getting help articles: {e}")
            return []

    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get market data"""
        try:
            data = await self.db.market_data.find_one({"symbol": symbol})
            return MarketData(**data) if data else None
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None

    async def update_market_data(self, symbol: str, update_data: Dict) -> bool:
        """Update market data"""
        try:
            result = await self.db.market_data.update_one(
                {"symbol": symbol},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
            return False

    async def get_technical_indicators(self, symbol: str) -> List[TechnicalIndicator]:
        """Get technical indicators"""
        try:
            cursor = self.db.technical_indicators.find({"symbol": symbol})
            indicators = await cursor.to_list(length=None)
            return [TechnicalIndicator(**i) for i in indicators]
        except Exception as e:
            logger.error(f"Error getting technical indicators: {e}")
            return []

    async def get_orb_levels(self, symbol: str) -> Optional[ORBLevels]:
        """Get ORB levels"""
        try:
            data = await self.db.orb_levels.find_one({"symbol": symbol})
            return ORBLevels(**data) if data else None
        except Exception as e:
            logger.error(f"Error getting ORB levels: {e}")
            return None

    async def get_chat_history(self, user_id: int, limit: int = 10) -> List[ChatHistory]:
        """Get chat history"""
        try:
            cursor = self.db.chat_history.find(
                {"user_id": user_id}
            ).sort("timestamp", -1).limit(limit)
            history = await cursor.to_list(length=None)
            return [ChatHistory(**h) for h in history]
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []

    async def add_chat_message(self, message_data: Dict) -> Optional[ChatHistory]:
        """Add chat message"""
        try:
            message = ChatHistory(**message_data)
            await self.db.chat_history.insert_one(message.dict())
            return message
        except Exception as e:
            logger.error(f"Error adding chat message: {e}")
            return None

    async def get_user_feedback(self, user_id: int) -> List[UserFeedback]:
        """Get user feedback"""
        try:
            cursor = self.db.user_feedback.find({"user_id": user_id})
            feedback = await cursor.to_list(length=None)
            return [UserFeedback(**f) for f in feedback]
        except Exception as e:
            logger.error(f"Error getting user feedback: {e}")
            return []

    async def add_user_feedback(self, feedback_data: Dict) -> Optional[UserFeedback]:
        """Add user feedback"""
        try:
            feedback = UserFeedback(**feedback_data)
            await self.db.user_feedback.insert_one(feedback.dict())
            return feedback
        except Exception as e:
            logger.error(f"Error adding user feedback: {e}")
            return None

    async def get_payout_requests(self, user_id: int) -> List[PayoutRequest]:
        """Get payout requests"""
        try:
            cursor = self.db.payout_requests.find({"user_id": user_id})
            requests = await cursor.to_list(length=None)
            return [PayoutRequest(**r) for r in requests]
        except Exception as e:
            logger.error(f"Error getting payout requests: {e}")
            return []

    async def create_payout_request(self, request_data: Dict) -> Optional[PayoutRequest]:
        """Create payout request"""
        try:
            request = PayoutRequest(**request_data)
            await self.db.payout_requests.insert_one(request.dict())
            return request
        except Exception as e:
            logger.error(f"Error creating payout request: {e}")
            return None

    async def get_market_scanner(self, user_id: int) -> Optional[MarketScanner]:
        """Get market scanner settings"""
        try:
            data = await self.db.market_scanner.find_one({"user_id": user_id})
            return MarketScanner(**data) if data else None
        except Exception as e:
            logger.error(f"Error getting market scanner: {e}")
            return None

    async def update_market_scanner(self, user_id: int, update_data: Dict) -> bool:
        """Update market scanner settings"""
        try:
            result = await self.db.market_scanner.update_one(
                {"user_id": user_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating market scanner: {e}")
            return False

    async def get_notifications(self, user_id: int) -> List[Notification]:
        """Get user notifications"""
        try:
            cursor = self.db.notifications.find({"user_id": user_id})
            notifications = await cursor.to_list(length=None)
            return [Notification(**n) for n in notifications]
        except Exception as e:
            logger.error(f"Error getting notifications: {e}")
            return []

    async def create_notification(self, notification_data: Dict) -> Optional[Notification]:
        """Create notification"""
        try:
            notification = Notification(**notification_data)
            await self.db.notifications.insert_one(notification.dict())
            return notification
        except Exception as e:
            logger.error(f"Error creating notification: {e}")
            return None

    async def get_session(self, token: str) -> Optional[Session]:
        """Get session by token"""
        try:
            data = await self.db.sessions.find_one({"token": token})
            return Session(**data) if data else None
        except Exception as e:
            logger.error(f"Error getting session: {e}")
            return None

    async def create_session(self, session_data: Dict) -> Optional[Session]:
        """Create new session"""
        try:
            session = Session(**session_data)
            await self.db.sessions.insert_one(session.dict())
            return session
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None

    async def delete_session(self, token: str) -> bool:
        """Delete session"""
        try:
            result = await self.db.sessions.delete_one({"token": token})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False

    async def log_error(self, error_data: Dict) -> Optional[ErrorLog]:
        """Log error"""
        try:
            error = ErrorLog(**error_data)
            await self.db.error_logs.insert_one(error.dict())
            return error
        except Exception as e:
            logger.error(f"Error logging error: {e}")
            return None

    async def get_api_rate_limit(self, endpoint: str, user_id: int) -> Optional[APIRateLimit]:
        """Get API rate limit"""
        try:
            data = await self.db.api_rate_limits.find_one({
                "endpoint": endpoint,
                "user_id": user_id
            })
            return APIRateLimit(**data) if data else None
        except Exception as e:
            logger.error(f"Error getting API rate limit: {e}")
            return None

    async def update_api_rate_limit(self, endpoint: str, user_id: int, update_data: Dict) -> bool:
        """Update API rate limit"""
        try:
            result = await self.db.api_rate_limits.update_one(
                {
                    "endpoint": endpoint,
                    "user_id": user_id
                },
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating API rate limit: {e}")
            return False

    async def create_collections(self):
        """Stub for creating collections (does nothing)."""
        logger.info("Stub: create_collections called.")
        return True 