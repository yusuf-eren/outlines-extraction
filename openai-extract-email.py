import outlines
from outlines.types import Regex
from pydantic import BaseModel
from openai import OpenAI

# Define the output schema using Pydantic
class EmailExtraction(BaseModel):
    name: str
    surname: str
    email: str
    phone: str

# Create an OpenAI client instance
openai_client = OpenAI()

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
print("Extracted email:", result) # Extracted email: {"email":"erenyusuf170@gmail.com"}