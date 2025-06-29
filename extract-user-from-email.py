import outlines
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# Template builder
def template(user_input: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.\n<|im_end|>\n"
        f"<|im_start|>user\n{user_input}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# Define schema using Pydantic
class ContactInfo(BaseModel):
    name: str
    surname: str
    phone: str
    user_id: str


# Load model
hf_model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)
hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

model = outlines.from_transformers(hf_model, hf_tokenizer)

# Prompt with structured data
email_prompt = template(
    "Hi, my name is Yusuf Eren. You can reach me at 0555-123-4567 or with user ID 98yx01. Cheers!"
)

# Run extraction
result = model(email_prompt, output_type=ContactInfo, max_new_tokens=100)

# Display raw result + parsed
print("Raw JSON string:", result)
print("Parsed:", ContactInfo.model_validate_json(result))

# Raw JSON string: { "name": "Yusuf Eren", "surname": "Eren", "phone": "0555-123-4567", "user_id": "98yx01" }
# Parsed: name='Yusuf Eren' surname='Eren' phone='0555-123-4567' user_id='98yx01'
