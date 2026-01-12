import os
import dotenv

dotenv.load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")

SMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
GAODE_API_KEY = os.getenv("GAODE_API_KEY")
