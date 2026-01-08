import os
import dotenv

dotenv.load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")

SMITH_API_KEY = os.getenv("SMITH_API_KEY")
