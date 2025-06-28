import outlines
from outlines.samplers import greedy


def template(user_input: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# Initialize model
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = outlines.models.transformers(model_name)

# Email regex pattern for extraction
email_regex = r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,10}"

# Prompt template
email_prompt = template(
    "Hi John,Thanks for reaching out. You can email me at erenyusuf170@gmail.com anytime.Best,Yusuf"
)

email_generator = outlines.generate.regex(model, email_regex, sampler=greedy())

result = email_generator(email_prompt)

print("Result:", result)
# Result: erenyusuf170@gmail.com
