"""
Example: Summarize a customer support ticket using Groq's OpenAI-compatible API.
Loads the API key from a .env file for security and best practice.
Compatible with openai>=1.0.0 (v1 API).
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file. Please add GROQ_API_KEY=your_key to .env.")

client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

prompt = (
    "Summarize the following customer support ticket in 2 sentences. "
    "Focus on the main issue and requested resolution.\n"
    "Ticket: {ticket_text}"
)

ticket_text = "Customer cannot log in after password reset. Error message: 'Invalid credentials.' Tried resetting password again, but still cannot access account. Needs urgent resolution."

response = client.chat.completions.create(
    model="deepseek-r1-distill-llama-70b",
    messages=[
        {"role": "user", "content": prompt.format(ticket_text=ticket_text)}
    ],
    temperature=0.2,
    max_tokens=100,
)

print("Summary:")
print(response.choices[0].message.content) 