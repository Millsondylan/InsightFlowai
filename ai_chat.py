import google.generativeai as genai
from typing import Dict, List, Optional
import logging
from datetime import datetime
from config import GOOGLE_AI_API_KEY, AI_CHAT_SETTINGS
from database import Database
from utils import format_error, format_success

logger = logging.getLogger(__name__)

class AIChatHandler:
    """Handles AI chat interactions"""
    
    def __init__(self, db: Database):
        self.db = db
        self.model = None
        self.chat = None
        self.initialize_ai()

    def initialize_ai(self):
        """Initialize Google AI model"""
        try:
            genai.configure(api_key=GOOGLE_AI_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
            self.chat = self.model.start_chat(history=[])
            logger.info("AI model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AI model: {e}")
            raise

    async def get_chat_history(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get user's chat history"""
        try:
            return await self.db.get_chat_history(user_id, limit)
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []

    async def add_chat_message(self, user_id: int, message: str, is_user: bool = True) -> Optional[Dict]:
        """Add message to chat history"""
        try:
            message_data = {
                'user_id': user_id,
                'message': message,
                'is_user': is_user,
                'timestamp': datetime.now()
            }
            return await self.db.add_chat_message(message_data)
        except Exception as e:
            logger.error(f"Error adding chat message: {e}")
            return None

    async def process_message(self, user_id: int, message: str) -> str:
        """Process user message and generate AI response"""
        try:
            # Add user message to history
            await self.add_chat_message(user_id, message, True)
            
            # Get chat history for context
            history = await self.get_chat_history(user_id)
            context = self._prepare_context(history)
            
            # Generate AI response
            response = await self._generate_response(message, context)
            
            # Add AI response to history
            await self.add_chat_message(user_id, response, False)
            
            return response
        except Exception as e:
            error_msg = f"Error processing message: {e}"
            logger.error(error_msg)
            return format_error(error_msg)

    def _prepare_context(self, history: List[Dict]) -> str:
        """Prepare chat history as context"""
        try:
            context = "Previous conversation:\n"
            for msg in history:
                role = "User" if msg['is_user'] else "AI"
                context += f"{role}: {msg['message']}\n"
            return context
        except Exception as e:
            logger.error(f"Error preparing context: {e}")
            return ""

    async def _generate_response(self, message: str, context: str) -> str:
        """Generate AI response"""
        try:
            prompt = f"{context}\nUser: {message}\nAI:"
            response = await self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            logger.error(error_msg)
            return format_error(error_msg)

    async def analyze_market_data(self, market_data: Dict) -> str:
        """Analyze market data and provide insights"""
        try:
            prompt = f"""
            Analyze the following market data and provide insights:
            Symbol: {market_data['symbol']}
            Price: {market_data['price']}
            Change: {market_data['change']} ({market_data['change_percent']}%)
            Volume: {market_data['volume']}
            
            Please provide:
            1. Technical analysis
            2. Key support/resistance levels
            3. Trading recommendations
            4. Risk assessment
            """
            
            response = await self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            error_msg = f"Error analyzing market data: {e}"
            logger.error(error_msg)
            return format_error(error_msg)

    async def analyze_portfolio(self, portfolio: List[Dict]) -> str:
        """Analyze portfolio and provide recommendations"""
        try:
            portfolio_summary = "\n".join([
                f"Symbol: {item['symbol']}, Quantity: {item['quantity']}, "
                f"Entry: {item['entry_price']}, Current: {item['current_price']}"
                for item in portfolio
            ])
            
            prompt = f"""
            Analyze the following portfolio and provide recommendations:
            {portfolio_summary}
            
            Please provide:
            1. Portfolio performance analysis
            2. Risk assessment
            3. Diversification analysis
            4. Recommendations for rebalancing
            """
            
            response = await self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            error_msg = f"Error analyzing portfolio: {e}"
            logger.error(error_msg)
            return format_error(error_msg)

    async def explain_technical_indicator(self, indicator: str, data: Dict) -> str:
        """Explain technical indicator and its implications"""
        try:
            prompt = f"""
            Explain the following technical indicator and its implications:
            Indicator: {indicator}
            Value: {data['value']}
            Timeframe: {data['timeframe']}
            
            Please provide:
            1. What this indicator means
            2. How to interpret the current value
            3. Trading implications
            4. Limitations and considerations
            """
            
            response = await self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            error_msg = f"Error explaining technical indicator: {e}"
            logger.error(error_msg)
            return format_error(error_msg)

    async def provide_trading_education(self, topic: str) -> str:
        """Provide trading education on specific topic"""
        try:
            prompt = f"""
            Provide comprehensive education on the following trading topic:
            {topic}
            
            Please include:
            1. Basic concepts and definitions
            2. Key principles and strategies
            3. Common mistakes to avoid
            4. Practical examples
            5. Risk management considerations
            """
            
            response = await self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            error_msg = f"Error providing trading education: {e}"
            logger.error(error_msg)
            return format_error(error_msg)

    async def analyze_news_impact(self, news: List[Dict], symbol: str) -> str:
        """Analyze news impact on specific symbol"""
        try:
            news_summary = "\n".join([
                f"Title: {item['title']}\nDescription: {item['description']}\n"
                for item in news
            ])
            
            prompt = f"""
            Analyze the impact of the following news on {symbol}:
            {news_summary}
            
            Please provide:
            1. News sentiment analysis
            2. Potential market impact
            3. Trading implications
            4. Risk factors to consider
            """
            
            response = await self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            error_msg = f"Error analyzing news impact: {e}"
            logger.error(error_msg)
            return format_error(error_msg)

    async def provide_risk_analysis(self, trade_data: Dict) -> str:
        """Provide risk analysis for potential trade"""
        try:
            prompt = f"""
            Analyze the risk for the following potential trade:
            Symbol: {trade_data['symbol']}
            Entry Price: {trade_data['entry_price']}
            Stop Loss: {trade_data['stop_loss']}
            Take Profit: {trade_data['take_profit']}
            Position Size: {trade_data['position_size']}
            Account Balance: {trade_data['account_balance']}
            
            Please provide:
            1. Risk/reward analysis
            2. Position sizing assessment
            3. Risk management recommendations
            4. Alternative strategies
            """
            
            response = await self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            error_msg = f"Error providing risk analysis: {e}"
            logger.error(error_msg)
            return format_error(error_msg)

    async def explain_market_conditions(self, market_data: Dict) -> str:
        """Explain current market conditions"""
        try:
            prompt = f"""
            Explain the current market conditions based on the following data:
            Market: {market_data['market']}
            Trend: {market_data['trend']}
            Volatility: {market_data['volatility']}
            Volume: {market_data['volume']}
            Key Levels: {market_data['key_levels']}
            
            Please provide:
            1. Market overview
            2. Key factors affecting the market
            3. Trading opportunities
            4. Risk considerations
            """
            
            response = await self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            error_msg = f"Error explaining market conditions: {e}"
            logger.error(error_msg)
            return format_error(error_msg)

    async def provide_strategy_analysis(self, strategy: str, parameters: Dict) -> str:
        """Analyze trading strategy"""
        try:
            prompt = f"""
            Analyze the following trading strategy:
            Strategy: {strategy}
            Parameters: {parameters}
            
            Please provide:
            1. Strategy overview
            2. Performance analysis
            3. Risk assessment
            4. Optimization suggestions
            5. Implementation guidelines
            """
            
            response = await self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            error_msg = f"Error providing strategy analysis: {e}"
            logger.error(error_msg)
            return format_error(error_msg) 