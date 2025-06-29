import os
from dotenv import load_dotenv
import outlines
from outlines.types import Regex
from pydantic import BaseModel
from openai import OpenAI

# .env dosyasını yükle
load_dotenv()

# API key'i .env dosyasından çek
api_key = os.getenv('OPENAI_API_KEY')

# API key boş mu kontrol et
if not api_key:
    raise ValueError("OPENAI_API_KEY .env dosyasında tanımlı değil")

# OpenAI client'ını API key ile oluştur
openai_client = OpenAI(api_key=api_key)

# Define the output schema using Pydantic
class EmailExtraction(BaseModel):
    name: str
    surname: str
    email: str
    phone: str

# Create an Outlines model
model = outlines.from_openai(openai_client, "gpt-4o")

# Text containing an email
text = """
Hi John,

Thanks for reaching out. You can email me at erenyusuf170@gmail.com anytime.

Best,
Yusuf

+1234567890
"""

result = model(text, EmailExtraction)
print("Extracted email:", result)
