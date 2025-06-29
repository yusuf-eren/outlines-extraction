import outlines
from outlines.types import Regex
from transformers import AutoModelForCausalLM, AutoTokenizer


def template(user_input: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# Initialize model
hf_model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)
hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

model = outlines.from_transformers(hf_model, hf_tokenizer)

# Email regex pattern for extraction
email_regex = Regex(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,10}")

# Prompt template
email_prompt = template(
    "Hi John,Thanks for reaching out. You can email me at erenyusuf170@gmail.com anytime.Best,Yusuf."
)


result = model(email_prompt, email_regex)

print("Result:", result)
# Result: erenyusuf170@gmail.com
