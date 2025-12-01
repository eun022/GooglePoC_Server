from google import genai
from config.gemini_api import API_KEY

client = genai.Client(api_key=API_KEY)
__all__ = ["client"]



