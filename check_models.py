from google import genai

# Nhớ điền API Key của ông vào đây
GEMINI_API_KEY = "AIzaSyD_X9LYKC57PCxnhSmDmvm5Lw7GJVDS1mQ" 

client = genai.Client(api_key=GEMINI_API_KEY)

print("🔍 ĐANG QUÉT DANH SÁCH MODEL KHẢ DỤNG...")
for model in client.models.list():
    print(f"- {model.name}")