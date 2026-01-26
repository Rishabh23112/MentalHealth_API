from abc import ABC, abstractmethod

class AlertProvider(ABC):
    @abstractmethod
    def send_alert(self, data: dict) -> bool:
        """
        Sends an alert.
        :param data: Dictionary containing alert details (message, recipient, etc.)
        :return: True if successful, False otherwise.
        """
        pass
