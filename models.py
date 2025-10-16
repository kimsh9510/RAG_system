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
    _ensure_packages(["transformers==4.42.3", "accelerate", "bitsandbytes"])
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

def load_solar_pro2(model_id: str = "solar-pro2", reasoning_effort: str = "minimal", max_new_tokens: int = 1024, temperature: float = 0.7):
    """
    Initialize Solar Pro 2 model via the Upstage API (langchain-upstage wrapper).

    Returns a small client with:
      - invoke(prompt: str) -> str
      - with_structured_output(json_schema: dict) -> object with invoke(prompt: str) -> dict

    The function reads the API key from the environment variable 'UPSTAGE_API_KEY'.
    If you store the token under a different name, either set UPSTAGE_API_KEY or update this function.
    """
    # ensure the required packages are installed before importing
    _ensure_packages(["langchain-core", "langchain-upstage"])

    # lazy import to avoid hard dependency at module import time
    from langchain_upstage import ChatUpstage
    try:
        # langchain_core.messages moved between releases; import guard
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:
        # fallback import path if package provides compatible names
        try:
            from langchain_core.schema import HumanMessage, SystemMessage
        except Exception:
            # create tiny stand-in dataclass for messages if import fails
            class HumanMessage:
                def __init__(self, content: str):
                    self.content = content

            class SystemMessage(HumanMessage):
                pass

    load_dotenv()
    token = os.environ.get("UPSTAGE_API_KEY") or os.environ.get("SOLAR_PRO_API_KEY") or os.environ.get("HF_HUB_TOKEN")
    if not token:
        raise RuntimeError("Solar Pro 2 API key not found. Please set UPSTAGE_API_KEY (or SOLAR_PRO_API_KEY) in your .env file.")

    chat = ChatUpstage(api_key=token, model=model_id, reasoning_effort=reasoning_effort)

    class SolarPro2Client:
        def __init__(self, chat, max_new_tokens=max_new_tokens, temperature=temperature):
            self.chat = chat
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature

        def invoke(self, prompt: str, *, reasoning_effort: str | None = None, response_format: dict | None = None, **kwargs):
            """
            Invoke the Solar Pro 2 model with a simple prompt string.
            Returns a string for text responses; if the underlying client returns a dict it will
            attempt to extract the assistant's content.
            Additional completion kwargs are passed through to the underlying client.
            """
            messages = [HumanMessage(content=prompt)]
            # prefer per-call reasoning_effort if provided
            call_kwargs = {"max_tokens": self.max_new_tokens, "temperature": self.temperature}
            if reasoning_effort is not None:
                call_kwargs["reasoning_effort"] = reasoning_effort
            if response_format is not None:
                call_kwargs["response_format"] = response_format
            call_kwargs.update(kwargs)

            resp = self.chat.invoke(messages, **call_kwargs)

            # attempt to normalize the response into a string (compat with existing code)
            try:
                if isinstance(resp, dict):
                    # standard chat completion structure: choices[0].message.content
                    choices = resp.get("choices") if isinstance(resp, dict) else None
                    if choices and isinstance(choices, list) and len(choices) > 0:
                        first = choices[0]
                        message = first.get("message") if isinstance(first, dict) else None
                        if isinstance(message, dict):
                            content = message.get("content")
                            if content is not None:
                                return content
                        # sometimes the library returns an assistant string directly
                        if isinstance(first.get("message"), str):
                            return first.get("message")
                    # fallback: try common top-level fields
                    if "message" in resp and isinstance(resp["message"], str):
                        return resp["message"]
                    # if the model returned structured data (dict), return the dict as string for compatibility
                    return str(resp)
                else:
                    return str(resp)
            except Exception:
                return str(resp)

        def with_structured_output(self, json_schema: dict):
            """
            Return a structured-output wrapper which will request the model to follow the provided JSON Schema.
            The returned object exposes invoke(prompt: str) and will return the parsed structured result (likely a dict).
            """
            structured_llm = self.chat.with_structured_output({"type": "json_schema", "json_schema": json_schema})

            class StructuredWrapper:
                def __init__(self, structured_llm):
                    self.structured_llm = structured_llm

                def invoke(self, prompt: str, **kwargs):
                    messages = [HumanMessage(content=prompt)]
                    resp = self.structured_llm.invoke(messages, **kwargs)
                    return resp

            return StructuredWrapper(structured_llm)

    return SolarPro2Client(chat)

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