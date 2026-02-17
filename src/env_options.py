import sys
import os
import torch
import transformers
from typing import List, Dict

def check_env(colab:bool=False, use_dotenv:bool=True, dotenv_path:str=None, colab_secrets:dict=None, env_tokens:List[str]=None) -> Dict[str, str]:
    # Checking versions and GPU availability:
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"FlashAttention available: {torch.backends.cuda.flash_sdp_enabled()}")
    else:
        print("No CUDA device available")

    if use_dotenv:    
        from dotenv import load_dotenv
        load_dotenv(dotenv_path) # path to your dotenv file
        print(f"Retrieving token(s) from {dotenv_path} or environment variables")

    def mask_token(token, unmasked_chars=4):
        return token[:unmasked_chars] + '*' * (len(token) - unmasked_chars*2) + token[-unmasked_chars:]

    tokens = {}
    for token_name in env_tokens or []:
        if use_dotenv:
            token = os.getenv(token_name)
        elif colab:
            token = colab_secrets.get(token_name)
        else:
            token = os.environ.get(token_name)

        if token is None:
            print(f"{token_name} not found in the provided .env file or environment variables")
        else:
            print(f"Using {token_name}: {mask_token(token)}")
            tokens[token_name] = token

    return tokens
