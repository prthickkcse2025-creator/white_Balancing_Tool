from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Use a model you know you have access to
response = client.models.generate_content(
    model='gemini-2.0-flash', 
    contents="Hello, are you working?"
)
print(response.text)