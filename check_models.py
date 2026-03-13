from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# Nhớ điền API Key của ông vào đây
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

print("🔍 ĐANG QUÉT DANH SÁCH MODEL KHẢ DỤNG...")
for model in client.models.list():
    print(f"- {model.name}")