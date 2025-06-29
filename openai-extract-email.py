import os
from dotenv import load_dotenv
import outlines
from outlines.types import Regex
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("OPENAI_API_KEY is not defined in .env file")

# Create an OpenAI client instance
openai_client = OpenAI(api_key=api_key)

# Create an Outlines model
model = outlines.from_openai(openai_client, "gpt-4o")

class EmailExtraction(BaseModel):
    name: str
    surname: str
    email: str
    phone: str

# Text containing an email
text = """
Hi John,

Thanks for reaching out. You can email me at erenyusuf170@gmail.com anytime.

Best,
Yusuf

+1234567890
"""

result = model(text, EmailExtraction)
print("Extracted email:", result) # Extracted email: {"email":"erenyusuf170@gmail.com"}