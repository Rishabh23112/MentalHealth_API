import logging
import os
from .sms import TwilioProvider
from .telegram import TelegramProvider

logger = logging.getLogger(__name__)

class AlertDispatcher:
    def __init__(self):
        self.sms_provider = TwilioProvider()
        self.telegram_provider = TelegramProvider()
        self.helpline_number = os.getenv("HELPLINE_PHONE_NUMBER")

    def trigger_alert(self, user_name: str, reason: str, location: str, short_message: str):
        """
        Orchestrates the alert process:
        1. Try SMS to helpline/admin.
        2. If SMS fails, send backup alert to Telegram.
        """
        
        # Prepare Data
        sms_message = f"CRISIS ALERT: User {user_name} at {location}. Reason: {reason}. Msg: {short_message}"
        
        telegram_message = (
            f"🚨 *CRISIS DETECTED* 🚨\n\n"
            f"👤 *User:* {user_name}\n"
            f"📍 *Location:* {location}\n"
            f"⚠️ *Reason:* {reason}\n"
            f"💬 *Message:* {short_message}\n\n"
            f"Please take immediate action."
        )

        # 1. Try SMS
        sms_data = {
            "phone_number": self.helpline_number,
            "message": sms_message
        }
        
        if self.helpline_number:
            logger.info("Attempting to send SMS alert...")
            sms_success = self.sms_provider.send_alert(sms_data)
        else:
            logger.warning("HELPLINE_PHONE_NUMBER not set. Skipping SMS.")
            sms_success = False

        # 2. If SMS fails, try Telegram
        if not sms_success:
            logger.warning("SMS failed or skipped. Falling back to Telegram...")
            telegram_data = {
                "message": telegram_message
            }
            telegram_success = self.telegram_provider.send_alert(telegram_data)
            
            if not telegram_success:
                logger.critical("❌ CRITICAL: BOTH SMS AND TELEGRAM AIERTS FAILED.")
        else:
            logger.info("SMS delivered. Skipping backup Telegram alert.")
