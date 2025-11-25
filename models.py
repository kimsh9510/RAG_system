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


# ============================================================
# Embeddings
# ============================================================
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


# ============================================================
# HuggingFace Local Models (NO TRUNCATION)
# ============================================================
def _hf_pipeline(model, tokenizer, max_new_tokens):
    """Returns a HuggingFacePipeline WITHOUT truncation=True."""
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        truncation=False,        # 🔥 IMPORTANT FIX
        pad_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=gen)


def load_llm(model_id="Qwen/Qwen2-7B-Instruct", max_new_tokens=4096):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
    )
    return _hf_pipeline(model, tokenizer, max_new_tokens)


def load_llama3(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=4096):
    _ensure_packages(["transformers==4.42.3", "accelerate", "bitsandbytes"])

    load_dotenv()
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if token:
        os.environ["HF_HUB_TOKEN"] = token

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_auth_token=token,
    )

    return _hf_pipeline(model, tokenizer, max_new_tokens)


def load_solar_pro(model_id="upstage/solar-pro-preview-instruct", max_new_tokens=4096):
    _ensure_packages(["transformers==4.44.2", "torch==2.3.1", "flash_attn==2.5.8", "accelerate==0.31.0"])
    import gc

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    return _hf_pipeline(model, tokenizer, max_new_tokens)


# ============================================================
# EXAONE — Local
# ============================================================
def load_EXAONE(model_id="LGAI-EXAONE/EXAONE-4.0.1-32B", max_new_tokens=4096):
    _ensure_packages(["transformers>=4.54.0", "accelerate", "bitsandbytes"])
    import gc

    load_dotenv()
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_HUB_TOKEN")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_auth_token=token,
        trust_remote_code=True,
    )

    return _hf_pipeline(model, tokenizer, max_new_tokens)


# ============================================================
# Friendli EXAONE API (OpenAI-compatible)
# ============================================================
def load_EXAONE_api(model_id="dep9i05uqo2xp7u", max_new_tokens=4096, base_url="https://api.friendli.ai/dedicated"):
    _ensure_packages(["openai", "requests"])
    load_dotenv()

    token = os.environ.get("FRIENDLI_TOKEN") or os.environ.get("EXAONE_API_KEY")
    team = os.environ.get("FRIENDLI_TEAM")

    if not token:
        raise RuntimeError("Friendli/EXAONE API key not found.")

    # try SDK
    try:
        from openai import OpenAI
        sdk = OpenAI(api_key=token, base_url=base_url)
    except:
        sdk = None

    class EXAONEClient:
        def __init__(self):
            self.model = model_id
            self.max_tokens = max_new_tokens

        def invoke(self, prompt: str, **kwargs):
            # Try OpenAI SDK first
            if sdk:
                try:
                    resp = sdk.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=self.max_tokens,
                    )
                    return resp.choices[0].message.content
                except:
                    pass

            # Fallback to raw HTTP
            import requests
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "X-Friendli-Team": team,
            }
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.max_tokens,
            }
            resp = requests.post(
                f"{base_url}/v1/chat/completions", json=body, headers=headers
            )
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    return EXAONEClient()


# ============================================================
# gpt-5.1/ gpt-5-mini (OpenAI-compatible)
# ============================================================
def load_paid_gpt(
    model_id="gpt-5-mini", # or use gpt-5.1
    max_output_tokens=4096,
    api_key_env=None,
    base_url=None,
):
    import json
    from dotenv import load_dotenv
    load_dotenv()

    token = (
        os.environ.get(api_key_env) if api_key_env else None
    ) or os.environ.get("GPT_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if not token:
        raise RuntimeError("Missing GPT API key.")

    from openai import OpenAI
    client = OpenAI(api_key=token, base_url=base_url)

    USE_RESP = model_id.startswith(("gpt-5", "gpt-4.1", "gpt-4o"))

    class GPT:
        def __init__(self):
            self.model = model_id
            self.max_tokens = max_output_tokens

        # ======================================================
        # INVOKE
        # ======================================================
        def invoke(self, prompt: str, **kwargs):
            max_tokens = kwargs.pop("max_tokens", self.max_tokens)

            if USE_RESP:
                resp = client.responses.create(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=max_tokens,
                )

                # 🔥 FIXED — merge ALL output chunks
                chunks = []
                for o in resp.output_text:
                    chunks.append(o)
                return "".join(chunks)

            # fallback → chat.completions
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content

    return GPT()

def load_gpt(
    model_id: str = "gpt-4o-mini",
    max_new_tokens: int = 512,
    api_key_env: str | None = None,
    base_url: str | None = None,
):
    """
    Load a GPT model accessed via API key (OpenAI-compatible). The function looks for the API key
    in the environment (.env) variables OPENAI_API_KEY or GPT_API_KEY by default, or the name
    provided via api_key_env. Returns a client with .invoke(prompt, **kwargs) and
    .with_structured_output(json_schema) for structured outputs.
    """
    _ensure_packages(["openai", "requests"])
    load_dotenv()

    # find API key
    key_names = [api_key_env] if api_key_env else []
    key_names += ["OPENAI_API_KEY"]
    token = None
    for k in key_names:
        if k and os.environ.get(k):
            token = os.environ.get(k)
            break

    if not token:
        raise RuntimeError(
            "GPT API key not found. Please set OPENAI_API_KEY or GPT_API_KEY (or pass api_key_env) in your .env file."
        )

    # try new OpenAI official client (OpenAI(api_key=...)) then fallback to legacy openai package usage
    OpenAIClient = None
    client_instance = None
    try:
        from openai import OpenAI as _OpenAIClient  # new SDK style
        OpenAIClient = _OpenAIClient
        client_instance = OpenAIClient(api_key=token, base_url=(base_url or None))
    except Exception:
        try:
            import openai as _openai_legacy
            _openai_legacy.api_key = token
            if base_url:
                _openai_legacy.api_base = base_url.rstrip("/")
            client_instance = _openai_legacy
        except Exception:
            client_instance = None

    class GPTApiClient:
        def __init__(self, client, model_id, max_new_tokens):
            self.client = client
            self.model_id = model_id
            self.max_new_tokens = max_new_tokens

        def _parse_resp_content(self, resp):
            # normalize typical responses to a string content
            try:
                # new SDK: resp.choices[0].message.content or resp.choices[0].message or resp.output[0].content
                if hasattr(resp, "choices"):
                    choices = getattr(resp, "choices")
                    if isinstance(choices, (list, tuple)) and len(choices) > 0:
                        first = choices[0]
                        if isinstance(first, dict):
                            msg = first.get("message") or {}
                            if isinstance(msg, dict):
                                return msg.get("content") or str(msg)
                            # legacy ChatCompletion style
                            if "text" in first:
                                return first["text"]
                        else:
                            # try dataclass-like objects
                            msg = getattr(first, "message", None) or getattr(first, "text", None)
                            if isinstance(msg, str):
                                return msg
                            if hasattr(msg, "content"):
                                return getattr(msg, "content")
                # new SDK direct chat response object
                if hasattr(resp, "choices") and len(resp.choices) > 0:
                    c = resp.choices[0]
                    if hasattr(c, "message") and hasattr(c.message, "content"):
                        return c.message.content
                # legacy fallback to dictionary/string
                if isinstance(resp, dict):
                    choices = resp.get("choices") or []
                    if choices:
                        msg = choices[0].get("message") or {}
                        content = msg.get("content")
                        if content is not None:
                            return content
                    return str(resp)
                return str(resp)
            except Exception:
                return str(resp)

        def invoke(self, prompt: str, **kwargs):
            """
            Invoke the model. kwargs may include max_tokens, temperature, etc.
            """
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

            params = {"model": self.model_id, "messages": messages, "max_tokens": kwargs.pop("max_tokens", self.max_new_tokens)}
            params.update(kwargs)

            # Try new OpenAI client first (object with .chat.create)
            try:
                if OpenAIClient is not None and hasattr(self.client, "chat"):
                    resp = self.client.chat.create(**params)
                    return self._parse_resp_content(resp)
            except Exception:
                pass

            # Fallback to legacy openai.ChatCompletion.create
            try:
                if hasattr(self.client, "ChatCompletion") or hasattr(self.client, "chat"):
                    # legacy openai: openai.ChatCompletion.create(...)
                    if hasattr(self.client, "ChatCompletion") and hasattr(self.client.ChatCompletion, "create"):
                        resp = self.client.ChatCompletion.create(**params)
                        return self._parse_resp_content(resp)
                    # some shims expose chat.create
                    if hasattr(self.client, "chat") and hasattr(self.client.chat, "create"):
                        resp = self.client.chat.create(**params)
                        return self._parse_resp_content(resp)
                # final http fallback using requests (very minimal)
                import requests
                headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
                url = (base_url.rstrip("/") if base_url else "https://api.openai.com") + "/v1/chat/completions"
                resp = requests.post(url, headers=headers, json=params, timeout=60)
                resp.raise_for_status()
                return self._parse_resp_content(resp.json())
            except Exception as e:
                raise RuntimeError(f"Failed to call GPT API: {e}") from e

        def with_structured_output(self, json_schema: dict):
            """
            Return a wrapper that attempts to ask the model to produce JSON conforming to json_schema.
            This uses response_format if available, otherwise prefixes an instruction and parses JSON.
            """
            def invoke_structured(prompt: str, **kwargs):
                # prefer provider-specific response_format if supported
                instruct = (
                    "You are to produce ONLY a JSON object that strictly follows the provided JSON Schema. "
                    "Do not add any commentary.\n\n"
                )
                # try with response_format param for new sdk/compatible APIs
                params = {
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": "You are a JSON API returning strictly formatted JSON."},
                        {"role": "user", "content": instruct + prompt},
                    ],
                    "max_tokens": kwargs.pop("max_tokens", self.max_new_tokens),
                    "response_format": {"type": "json_schema", "json_schema": {"schema": json_schema}},
                }
                params.update(kwargs)

                # try SDK first
                try:
                    if OpenAIClient is not None and hasattr(self.client, "chat"):
                        resp = self.client.chat.create(**params)
                        # some providers return parsed structure in choices[0].message.content or parsed field
                        parsed = self._parse_resp_content(resp)
                        # if parsed is string, try to json.loads
                        import json
                        if isinstance(parsed, str):
                            try:
                                return json.loads(parsed)
                            except Exception:
                                return parsed
                        return parsed
                except Exception:
                    pass

                # fallback: instruct + parse JSON from text response
                resp_text = self.invoke(instruct + prompt, **kwargs)
                import json
                try:
                    return json.loads(resp_text)
                except Exception:
                    return resp_text

            class StructuredWrapper:
                def __init__(self, fn):
                    self._fn = fn

                def invoke(self, prompt: str, **kwargs):
                    return self._fn(prompt, **kwargs)

            return StructuredWrapper(invoke_structured)

    return GPTApiClient(client_instance, model_id=model_id, max_new_tokens=max_new_tokens)