from google import genai
import os
from dotenv import load_dotenv

# Load your key from the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize client
client = genai.Client(api_key=api_key)

# Print models
for model in client.models.list():
    print(model.name)