
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameLogger:
    def __init__(self):
        self.history = []

    def log_action(self, action, result):
        self.history.append((action, result))
        logger.info(f"Action: {action}, Result: {result}")

    def log_inference(self, inference_result):
        logger.info(f"Inference Result: {inference_result}")
        self.history.append(("Inference", inference_result))

    def show_logs(self):
        for entry in self.history:
            print(entry)
