#LLM 모델 로드
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import langchain_core
import torch
import os
import subprocess
import sys
import importlib
import importlib.util

#for the package installation
def ensure_installed(packages: dict):
    for pkg, version in packages.items():
        subprocess.run([sys.executable, "-m", "pip", "install", f"{pkg}=={version}"], check=True)

def _ensure_packages(packages: list[str]):
    for pkg in packages:
        name = pkg.split("==")[0].split(">")[0]
        if importlib.util.find_spec(name) is None:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                importlib.invalidate_caches()
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to install {pkg}: {e}") from e
            
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64}
    )

def load_qwen():
    model_id = "Qwen/Qwen2-7B-Instruct"
    #model_id = "Qwen/Qwen2-1.5B-Instruct" #소형모델
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
    )
    pipe = pipeline("text-generation", 
                    model=model, 
                    tokenizer=tokenizer, 
                    max_length=2048,
                    max_new_tokens=1024,
                    truncation=True)
    return HuggingFacePipeline(pipeline=pipe)

# The solar-pro model needs following packages:
# pip install transformers==4.44.2 torch==2.3.1 flash_attn==2.5.8 accelerate==0.31.0
def load_solar_pro(model_id: str = "upstage/solar-pro-preview-instruct", max_new_tokens: int = 512):
    from transformers import BitsAndBytesConfig
    import gc

    required = {
        "transformers": "4.44.2",
        "torch": "2.3.1",
        "flash_attn": "2.5.8",
        "accelerate": "0.31.0"
    }

    for pkg, ver in required.items():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f"{pkg}=={ver}", "--quiet"],
            check=True
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Try 8-bit, then 4-bit, then full load
    model = None
    try:
        q8 = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16,
            trust_remote_code=True, quantization_config=q8, low_cpu_mem_usage=True
        )
    except Exception:
        try:
            q4 = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
            )
            torch.cuda.empty_cache()
            gc.collect()
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.float16,
                trust_remote_code=True, quantization_config=q4, low_cpu_mem_usage=True
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

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

# The llama3 model needs following pacakges:
# pip install --upgrade transformers accelerate bitsandbytes
def load_llama3(model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens: int = 512):
    load_dotenv()
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_HUB_TOKEN")
    if token:
        # ensure HF hub functions see the token early
        os.environ["HF_HUB_TOKEN"] = token

    # prefer bfloat16 when available for this model
    torch_dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16

    # Load tokenizer and model explicitly with the auth token so transformers can access gated repo.
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch_dtype,
        use_auth_token=token,
    )

    # Create pipeline using the already-loaded model/tokenizer (do NOT pass use_auth_token here).
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        truncation=True,
    )

    return HuggingFacePipeline(pipeline=gen)

# The EXAONE model needs following pacakges:
# pip install --upgrade transformers accelerate bitsandbytes
def load_EXAONE(model_name: str = "LGAI-EXAONE/EXAONE-4.0-32B", reasoning: bool = False):
    from transformers import BitsAndBytesConfig
    import gc

    required_packages = {
        "transformers": ">=4.45.0",   # EXAONE 4.0 지원 최소 버전
        "accelerate": ">=0.33.0",
        "bitsandbytes": ">=0.44.0"
    }

    for pkg, ver in required_packages.items():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f"{pkg}{ver}", "--quiet", "--upgrade"],
            check=True
        )

    def _max_memory(reserve_gb: int = 4):
        if not torch.cuda.is_available():
            return {}
        out = {}
        for i in range(torch.cuda.device_count()):
            total_gb = int(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3))
            usable = max(0, total_gb - reserve_gb)
            out[i] = f"{usable}GB"
        return out

    # tokenizer with fallback
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    max_memory = _max_memory()

    model = None
    # try 4-bit
    try:
        q4 = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16,
            trust_remote_code=True, quantization_config=q4, low_cpu_mem_usage=True, max_memory=max_memory or None
        )
    except Exception:
        pass

    # try 8-bit
    if model is None:
        try:
            q8 = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype=torch.float16,
                trust_remote_code=True, quantization_config=q8, low_cpu_mem_usage=True, max_memory=max_memory or None
            )
        except Exception:
            pass

    # final fallback
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=(torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16),
            device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True, max_memory=max_memory or None
        )

    class EXAONEWrapper:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

        def invoke(self, prompt: str, reasoning: bool = False, max_new_tokens: int = 128, **kwargs) -> str:
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", enable_thinking=reasoning
            )
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

            gen_kwargs = {"max_new_tokens": max_new_tokens}
            if reasoning:
                gen_kwargs.update({"do_sample": True, "temperature": 0.6, "top_p": 0.95})
            else:
                gen_kwargs.update({"do_sample": False, "temperature": 0.3})

            gen_kwargs.update(kwargs)
            gen_kwargs.pop("presence_penalty", None)

            input_ids = input_ids.to(device)
            out = self.model.generate(input_ids, **gen_kwargs)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)

    return EXAONEWrapper(model, tokenizer)

def load_gpt(
    #model_id: str = "gpt-4o-mini",
    model_id: str = "gpt-4.1-mini",
    max_new_tokens: int = 2048,
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
    #key_names += ["OPENAI_API_KEY", "GPT_API_KEY"]
    key_names += ["GPT_API_KEY", "OPENAI_API_KEY"]
    
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

def load_paid_gpt(
    model_id: str = "gpt-5.1",
    max_output_tokens: int = 1024,
    #max_output_tokens: int = 4096,
    api_key_env: str | None = None,
    base_url: str | None = None,
):
    """
    Universal GPT loader supporting:
    - GPT-5.1, GPT-4.1, GPT-4o → Responses API
    - GPT-3.5 / old models → ChatCompletions API fallback

    Returns:
        client.invoke(prompt)
        client.with_structured_output(schema).invoke(prompt)
    """
    import os, json, requests
    from dotenv import load_dotenv
    load_dotenv()

    # -------------------------
    # Load API Key
    # -------------------------
    key_names = [api_key_env] if api_key_env else []
    key_names += ["GPT_API_KEY"]
    token = None
    for k in key_names:
        if k and os.environ.get(k):
            token = os.environ[k]
            break
    if not token:
        raise RuntimeError("Missing API key: set OPENAI_API_KEY or GPT_API_KEY.")

    # -------------------------
    # Load OpenAI client (new SDK)
    # -------------------------
    try:
        from openai import OpenAI
        client = OpenAI(api_key=token, base_url=base_url)
    except Exception:
        client = None

    # Detect whether model requires Responses API
    USE_RESPONSES = model_id.startswith(("gpt-5", "gpt-4.1", "gpt-4o"))

    class GPTApiClient:
        def __init__(self):
            self.model = model_id
            self.max_tokens = max_output_tokens

        # ------------------------------------------------------------------
        # Unified INVOKE method
        # ------------------------------------------------------------------
        def invoke(self, prompt: str, **kwargs):
            """
            Main call: automatically picks correct API.
            """
            max_tokens = kwargs.pop("max_tokens", self.max_tokens)

            # ==============================================================
            # NEW: Use Responses API for GPT-5.1 / GPT-4.1 / GPT-4o
            # ==============================================================
            if USE_RESPONSES:
                try:
                    resp = client.responses.create(
                        model=self.model,
                        input=prompt,
                        max_output_tokens=max_tokens,
                        **kwargs
                    )
                    return resp.output_text
                except Exception as e:
                    raise RuntimeError(f"Responses API failed: {e}")

            # ==============================================================
            # FALLBACK: old ChatCompletion API (older models)
            # ==============================================================
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    **kwargs
                )
                return resp.choices[0].message.content

            except Exception as e:
                raise RuntimeError(f"ChatCompletion API failed: {e}")

        # ------------------------------------------------------------------
        # STRUCTURED OUTPUT SUPPORT
        # ------------------------------------------------------------------
        def with_structured_output(self, json_schema: dict):
            """
            Wrapper requiring JSON output according to a schema.
            Uses Responses API.
            """

            def _call(prompt, **kwargs):
                max_tokens = kwargs.pop("max_tokens", self.max_tokens)

                try:
                    resp = client.responses.create(
                        model=self.model,
                        input=prompt,
                        max_output_tokens=max_tokens,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"schema": json_schema}
                        },
                        **kwargs
                    )
                    return resp.output[0].content[0].text  # JSON string

                except Exception as e:
                    raise RuntimeError(f"Structured output failed: {e}")

            class Wrapper:
                def invoke(self, p, **kw):
                    out = _call(p, **kw)
                    try:
                        return json.loads(out)
                    except:
                        return out

            return Wrapper()

    return GPTApiClient()