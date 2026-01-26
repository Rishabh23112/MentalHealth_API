import logging
import os
from twilio.rest import Client

logger = logging.getLogger(__name__)

class TwilioProvider:
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.messaging_service_sid = os.getenv("TWILIO_MESSAGING_SERVICE_SID")
        
        if not self.account_sid or not self.auth_token:
             logger.warning("⚠️ Twilio Credentials missing. SMS will fail.")

    def send_alert(self, data: dict) -> bool:
        """
        Sends SMS using Twilio.
        :param data: {"phone_number": str, "message": str}
        """
        phone_number = data.get("phone_number")
        message_body = data.get("message")

        if not self.account_sid or not self.auth_token:
            logger.error("❌ Cannot send SMS: Twilio credentials not set.")
            return False

        try:
            client = Client(self.account_sid, self.auth_token)
            
            logger.info(f"Sending Twilio SMS to {phone_number}...")
            message = client.messages.create(
                messaging_service_sid=self.messaging_service_sid,
                body=message_body,
                to=f"+91{phone_number}" if not phone_number.startswith("+") else phone_number
            )
            
            logger.info(f"✅ Twilio SMS Sent! SID: {message.sid}")
            return True

        except Exception as e:
            logger.error(f"❌ Twilio Send Failed: {e}")
            return False
