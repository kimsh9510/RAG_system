from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import torch
import sys
import subprocess
import importlib
import importlib.util
import os

def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64},
    )
def _ensure_packages(packages: list[str]):
    for pkg in packages:
        name = pkg.split("==")[0].split(">")[0]
        if importlib.util.find_spec(name) is None:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                importlib.invalidate_caches()
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to install {pkg}: {e}") from e

def load_llm(model_id: str = "Qwen/Qwen2-7B-Instruct", max_new_tokens: int = 512):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens, truncation=True)
    return HuggingFacePipeline(pipeline=gen)

def load_llama3(model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens: int = 512):
    _ensure_packages(["transformers>=4.44.0", "accelerate", "bitsandbytes"])
    load_dotenv()
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if token:
        os.environ["HF_HUB_TOKEN"] = token

    torch_dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch_dtype,
        use_auth_token=token,
    )    
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        truncation=True,
    )
    return HuggingFacePipeline(pipeline=gen)

def load_solar_pro(model_id: str = "upstage/solar-pro-preview-instruct", max_new_tokens: int = 512):
    _ensure_packages(["transformers==4.44.2", "torch==2.3.1", "flash_attn==2.5.8", "accelerate==0.31.0"])
    import gc

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        truncation=True,
        do_sample=True,
        temperature=0.7,
        pad_token_id=getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=pipe)

def load_EXAONE(model_id: str = "LGAI-EXAONE/EXAONE-4.0.1-32B", max_new_tokens: int = 512):
    _ensure_packages(["transformers>=4.54.0", "accelerate", "bitsandbytes"])
    import gc

    load_dotenv()
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if token:
        os.environ["HF_HUB_TOKEN"] = token

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    torch_dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch_dtype,
        use_auth_token=token,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        truncation=True,
    )
    return HuggingFacePipeline(pipeline=gen)