import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.crisis.detector import CrisisDetector
from src.alerts.dispatcher import AlertDispatcher

class TestCrisisSystem(unittest.TestCase):
    
    def test_detector_positive_regex(self):
        """Test regex fallback"""
        detector = CrisisDetector() 
        is_crisis, reason = detector.detect("I want to kill myself")
        self.assertTrue(is_crisis)
        self.assertIn("suicidal thoughts", reason)

    def test_detector_positive_semantic(self):
        """Test semantic match with mocked embeddings"""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.9, 0.2]]
        mock_embeddings.embed_query.return_value = [1.0, 0.0]
        
        detector = CrisisDetector(embeddings_model=mock_embeddings)
        detector.semantic_phrases = ["Mock phrase"]
        detector.phrase_embeddings = mock_embeddings.embed_documents(["Mock phrase"]) 
        
        is_crisis, reason = detector.detect("Some user text")
        self.assertTrue(is_crisis)
        self.assertIn("severe distress", reason)

    @patch('src.alerts.dispatcher.TwilioProvider')
    @patch('src.alerts.dispatcher.TelegramProvider')
    def test_dispatcher_context(self, MockTelegram, MockTwilio):
        """Test that user name and location are passed correctly"""
        mock_sms = MockTwilio.return_value
        
        dispatcher = AlertDispatcher()
        dispatcher.helpline_number = "1234567890" 
        mock_sms.send_alert.return_value = True 
        
        # Trigger with specific user context
        dispatcher.trigger_alert(
            user_name="Rishabh", 
            reason="Test Reason", 
            location="Pune", 
            short_message="Help me"
        )
        
        # Verify call args
        args, _ = mock_sms.send_alert.call_args
        data = args[0]
        
        self.assertIn("Rishabh", data["message"])
        self.assertIn("Pune", data["message"])
        print("✅ User Context Verification Passed")

if __name__ == '__main__':
    unittest.main()
