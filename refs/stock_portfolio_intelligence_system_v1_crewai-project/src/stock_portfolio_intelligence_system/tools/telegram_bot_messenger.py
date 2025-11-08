from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import requests
import json

class TelegramMessageInput(BaseModel):
    """Input schema for Telegram Bot Messenger Tool."""
    bot_token: str = Field(
        ...,
        description="Telegram bot token obtained from @BotFather"
    )
    chat_id: str = Field(
        ...,
        description="Chat ID or Channel ID where the message will be sent. For channels, use @channelname or -100channelid format"
    )
    message: str = Field(
        ...,
        description="The message to send. Supports multi-line text and Markdown formatting (bold: **text**, italic: *text*, code: `text`)"
    )

class TelegramBotMessenger(BaseTool):
    """Tool for sending messages to Telegram chats or channels via Telegram Bot API."""

    name: str = "telegram_bot_messenger"
    description: str = (
        "Sends messages to Telegram chats or channels using the Telegram Bot API. "
        "Supports Markdown formatting and handles both private chats and public channels. "
        "Useful for notifications, alerts, or automated messaging."
    )
    args_schema: Type[BaseModel] = TelegramMessageInput

    def _run(self, bot_token: str, chat_id: str, message: str) -> str:
        """
        Send a message to a Telegram chat or channel.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Target chat or channel ID
            message: Message content with Markdown formatting
            
        Returns:
            String indicating success or failure with details
        """
        try:
            # Validate inputs
            if not bot_token or not bot_token.strip():
                return "Error: Bot token cannot be empty"
            
            if not chat_id or not chat_id.strip():
                return "Error: Chat ID cannot be empty"
            
            if not message or not message.strip():
                return "Error: Message cannot be empty"
            
            # Prepare the API URL
            api_url = f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
            
            # Prepare the payload
            payload = {
                'chat_id': chat_id.strip(),
                'text': message.strip(),
                'parse_mode': 'Markdown'  # Enable Markdown formatting
            }
            
            # Set headers
            headers = {
                'Content-Type': 'application/json'
            }
            
            # Make the HTTP POST request
            response = requests.post(
                api_url, 
                json=payload, 
                headers=headers,
                timeout=30  # 30 second timeout
            )
            
            # Parse the response
            response_data = response.json()
            
            if response.status_code == 200 and response_data.get('ok', False):
                message_id = response_data.get('result', {}).get('message_id', 'unknown')
                return f"✅ Message sent successfully! Message ID: {message_id}"
            else:
                error_description = response_data.get('description', 'Unknown error')
                error_code = response_data.get('error_code', response.status_code)
                return f"❌ Failed to send message. Error {error_code}: {error_description}"
                
        except requests.exceptions.Timeout:
            return "❌ Failed to send message: Request timed out after 30 seconds"
        
        except requests.exceptions.ConnectionError:
            return "❌ Failed to send message: Network connection error"
        
        except requests.exceptions.HTTPError as e:
            return f"❌ Failed to send message: HTTP error - {str(e)}"
        
        except json.JSONDecodeError:
            return "❌ Failed to send message: Invalid response from Telegram API"
        
        except Exception as e:
            return f"❌ Failed to send message: Unexpected error - {str(e)}"