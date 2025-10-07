import os

def load_hf_token():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN. Set it in your terminal.")
    return token
