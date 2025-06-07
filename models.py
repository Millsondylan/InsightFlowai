from datetime import datetime
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field
from config import SUBSCRIPTION_PLANS

class User(BaseModel):
    """User model for storing user information"""
    user_id: int
    username: str
    created_at: datetime = Field(default_factory=datetime.now)
    subscription: Optional[Dict] = None
    settings: Dict = Field(default_factory=dict)
    language: str = "en"
    timezone: str = "UTC"
    last_active: datetime = Field(default_factory=datetime.now)

class Portfolio(BaseModel):
    """Portfolio model for tracking user's positions"""
    user_id: int
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    total_value: float
    daily_change: float
    last_updated: datetime = Field(default_factory=datetime.now)

class Alert(BaseModel):
    """Price alert model"""
    user_id: int
    symbol: str
    price: float
    condition: str  # 'above' or 'below'
    triggered: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    triggered_at: Optional[datetime] = None

class Subscription(BaseModel):
    """Subscription model"""
    user_id: int
    plan: str
    start_date: datetime
    end_date: datetime
    status: str  # 'active', 'expired', 'cancelled'
    price: float
    features: List[str]

class Affiliate(BaseModel):
    """Affiliate program model"""
    user_id: int
    referral_code: str
    referrals: List[Dict] = Field(default_factory=list)
    earnings: float = 0.0
    payout_history: List[Dict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

class Course(BaseModel):
    """Trading course model"""
    id: str
    title: str
    description: str
    duration: str
    level: str
    topics: List[str]
    video_url: str
    materials: List[str]
    created_at: datetime = Field(default_factory=datetime.now)

class HelpArticle(BaseModel):
    """Help article model"""
    id: str
    title: str
    category: str
    content: str
    tags: List[str]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class MarketData(BaseModel):
    """Market data model"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime = Field(default_factory=datetime.now)
    source: str

class TechnicalIndicator(BaseModel):
    """Technical indicator model"""
    symbol: str
    indicator: str
    value: float
    timestamp: datetime = Field(default_factory=datetime.now)
    timeframe: str

class ORBLevels(BaseModel):
    """Opening Range Breakout levels model"""
    symbol: str
    open_price: float
    high_price: float
    low_price: float
    current_price: float
    volume: int
    breakout_up: bool
    breakout_down: bool
    range: float
    range_percent: float
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatHistory(BaseModel):
    """AI chat history model"""
    user_id: int
    messages: List[Dict] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.now)

class UserFeedback(BaseModel):
    """User feedback model"""
    user_id: int
    article_id: str
    is_helpful: bool
    feedback_text: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class PayoutRequest(BaseModel):
    """Payout request model"""
    user_id: int
    amount: float
    status: str  # 'pending', 'approved', 'rejected'
    requested_at: datetime = Field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None

class MarketScanner(BaseModel):
    """Market scanner model"""
    user_id: int
    filters: Dict
    symbols: List[str]
    last_scan: datetime = Field(default_factory=datetime.now)
    results: List[Dict] = Field(default_factory=list)

class Notification(BaseModel):
    """Notification model"""
    user_id: int
    type: str  # 'alert', 'system', 'market', 'subscription'
    message: str
    read: bool = False
    created_at: datetime = Field(default_factory=datetime.now)

class Session(BaseModel):
    """User session model"""
    user_id: int
    token: str
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: datetime
    last_activity: datetime = Field(default_factory=datetime.now)

class ErrorLog(BaseModel):
    """Error logging model"""
    error_type: str
    message: str
    stack_trace: Optional[str] = None
    user_id: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class APIRateLimit(BaseModel):
    """API rate limiting model"""
    endpoint: str
    user_id: int
    request_count: int
    window_start: datetime = Field(default_factory=datetime.now)
    window_end: datetime 