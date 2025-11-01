import langchain_core
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
    _ensure_packages(["langchain-core", "langchain-upstage"]) #ensure the required packages are installed
    
    # lazy import to avoid hard dependency at module import time
    from langchain_upstage import ChatUpstage
    try:
        # langchain_core.messages moved between releases; import guard
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:
        # fallback import path if package provides compatible names
        try:
            from langchain_core.schema import HumanMessage, SystemMessage # pyright: ignore[reportMissingImports]
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

def load_EXAONE(model_id: str = "LGAI-EXAONE/EXAONE-4.0.1-32B", max_new_tokens: int = 10024):
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

def load_EXAONE_api(model_id: str = "dep9i05uqo2xp7u", max_new_tokens: int = 512, base_url: str = "https://api.friendli.ai/dedicated"):
    _ensure_packages(["openai", "requests"])  # ensure both SDK and requests are available

    load_dotenv()
    token = os.environ.get("FRIENDLI_TOKEN") or os.environ.get("EXAONE_API_KEY")
    team = os.environ.get("FRIENDLI_TEAM")
    if not token:
        raise RuntimeError("EXAONE API key not found. Please set FRIENDLI_TOKEN or EXAONE_API_KEY in your .env or environment.")

    # Try to use the OpenAI-compatible SDK (Friendli provides an OpenAI-compatible API surface)
    OpenAI = None
    try:
        from openai import OpenAI as _OpenAI
        OpenAI = _OpenAI
    except Exception:
        try:
            import openai as _openai_module
            OpenAI = getattr(_openai_module, "OpenAI", None)
        except Exception:
            OpenAI = None

    # For dedicated endpoints the path is under the dedicated prefix; when base_url is
    # https://api.friendli.ai/dedicated the REST path for chat completions is /v1/chat/completions
    endpoint_path = "/v1/chat/completions"

    class EXAONEApiClient:
        def __init__(self, token: str, model_id: str, max_new_tokens: int, base_url: str, team: str | None = None, sdk_client=None):
            self.token = token
            self.model_id = model_id
            self.max_new_tokens = max_new_tokens
            self.base_url = base_url.rstrip("/")
            self.team = team
            self.sdk_client = sdk_client

        def _http_request(self, body: dict):
            import requests
            url = f"{self.base_url}{endpoint_path}"
            headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
            if self.team:
                headers["X-Friendli-Team"] = self.team
            resp = requests.post(url, headers=headers, json=body, timeout=60)
            if resp.status_code == 401 or resp.status_code == 403:
                raise RuntimeError(f"Authentication failed when calling Friendli API: HTTP {resp.status_code}. Check FRIENDLI_TOKEN and team header.")
            resp.raise_for_status()
            return resp.json()

        def invoke(self, prompt: str, **kwargs):
            # messages per Friendli OpenAPI
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

            params = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": kwargs.pop("max_tokens", self.max_new_tokens),
            }
            params.update(kwargs)

            # 1) If SDK is available, try it first
            if self.sdk_client is not None:
                try:
                    completion = self.sdk_client.chat.completions.create(**params)
                    try:
                        return completion.choices[0].message.content
                    except Exception:
                        return str(completion)
                except Exception as e:
                    # If permission/auth error, surface a friendly message and fall back to HTTP
                    name = getattr(e, "__class__", type(e)).__name__
                    if name in ("PermissionDeniedError", "AuthenticationError"):
                        raise RuntimeError(
                            "Authentication with Friendli SDK failed. Check FRIENDLI_TOKEN, FRIENDLI_TEAM and that the token has inference permissions."
                        ) from e
                    # otherwise we'll try HTTP fallback below

            # 2) HTTP fallback
            try:
                resp = self._http_request(params)
                # expected structure: {choices: [{message: {content: "..."}}], ...}
                choices = resp.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    if content is not None:
                        return content
                # final fallback
                return str(resp)
            except Exception as e:
                raise

        def with_structured_output(self, json_schema: dict):
            # Use instruction + response_format where possible. Friendli supports response_format param.
            def invoke_structured(prompt: str, **kwargs):
                instruct = (
                    "You are to produce ONLY a JSON object that strictly follows the provided JSON Schema. "
                    "Do not add any commentary.\n\n"
                )
                body = {
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": "You are a JSON API returning strictly formatted JSON."},
                        {"role": "user", "content": instruct + prompt},
                    ],
                    "response_format": {"type": "json_schema", "json_schema": {"schema": json_schema}},
                    "max_tokens": kwargs.pop("max_tokens", self.max_new_tokens),
                }
                body.update(kwargs)

                # Try HTTP request (prefer explicit control)
                resp = self._http_request(body)
                # Friendli may include the parsed object or a content string
                choices = resp.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    # If server returned structured parse in a field, try to extract reasoning_content or parsed JSON
                    if isinstance(content, dict):
                        return content
                    # otherwise try to parse JSON string
                    import json
                    try:
                        return json.loads(content)
                    except Exception:
                        return content
                return resp

            class StructuredWrapper:
                def __init__(self, fn):
                    self._fn = fn

                def invoke(self, prompt: str, **kwargs):
                    return self._fn(prompt, **kwargs)

            return StructuredWrapper(invoke_structured)

    # instantiate client wrapper
    sdk_client = None
    if OpenAI is not None:
        try:
            sdk_client = OpenAI(api_key=token, base_url=base_url)
        except Exception:
            sdk_client = None

    return EXAONEApiClient(token=token, model_id=model_id, max_new_tokens=max_new_tokens, base_url=base_url, team=team, sdk_client=sdk_client)