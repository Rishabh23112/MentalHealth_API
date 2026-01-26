import requests
import os
import logging
from .base import AlertProvider

logger = logging.getLogger(__name__)

class TelegramProvider(AlertProvider):
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def send_alert(self, data: dict) -> bool:
        message = data.get("message") # Full detailed message for Telegram

        if not self.bot_token or not self.chat_id:
            logger.error("Telegram credentials missing (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)")
            return False

        try:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(self.base_url, json=payload)
            
            if response.status_code == 200:
                logger.info("✅ Telegram alert sent successfully")
                return True
            else:
                logger.error(f"❌ Telegram alert failed with {response.status_code}: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Telegram Exception: {e}")
            return False
