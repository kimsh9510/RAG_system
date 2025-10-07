#LLM 모델 로드
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
from langchain_community.llms import HuggingFacePipeline
import torch

def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64}
    )

def load_llm():
    model_id = "Qwen/Qwen2-7B-Instruct"
    #model_id = "Qwen/Qwen2-1.5B-Instruct" #소형모델
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, truncation=True)
    return HuggingFacePipeline(pipeline=pipe)

def load_solar_pro():
    """Load Solar Pro model (solar-pro-preview-instruct) from Upstage with memory optimization"""
    from transformers import BitsAndBytesConfig
    import gc
    
    # Clear GPU cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    tokenizer = AutoTokenizer.from_pretrained("upstage/solar-pro-preview-instruct")
    
    # Configure 8-bit quantization for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True  # Offload some layers to CPU if needed
    )
    
    try:
        # Try loading with 8-bit quantization first
        model = AutoModelForCausalLM.from_pretrained(
            "upstage/solar-pro-preview-instruct",
            device_map="auto",  # Let it automatically decide device placement
            torch_dtype=torch.float16,  # Use float16 instead of auto
            trust_remote_code=True,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
            max_memory={0: "20GB", 1: "20GB"}  # Reserve some memory for operations
        )
        print("✅ Solar Pro model loaded with 8-bit quantization")
        
    except torch.cuda.OutOfMemoryError:
        print("⚠️ 8-bit loading failed, trying 4-bit quantization...")
        # Clear cache and try 4-bit quantization
        torch.cuda.empty_cache()
        gc.collect()
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            "upstage/solar-pro-preview-instruct",
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            max_memory={0: "20GB", 1: "20GB"}
        )
        print("✅ Solar Pro model loaded with 4-bit quantization")
    
    # Create a pipeline for use with LangChain
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=512, 
        truncation=True,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    )
    return HuggingFacePipeline(pipeline=pipe)



def load_EXAONE(model_name: str = "LGAI-EXAONE/EXAONE-4.0-32B", reasoning: bool = False):
    """Load EXAONE model and tokenizer with recommended defaults.

    Args:
        model_name: HuggingFace model identifier.
        reasoning: If True, configures tokenizer for reasoning (<think> block).

    Returns:
        tuple: (model, tokenizer)

    Notes:
        - For non-reasoning mode we recommend temperature < 0.6 and do_sample=False.
        - For reasoning mode we recommend temperature=0.6 and top_p=0.95, do_sample=True.
        - If you want to enable presence penalty during generation, pass `presence_penalty` to your generation call
          (some wrappers expose this differently). The model.generate call below shows how to use the tokenizer's
          apply_chat_template utilities.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import gc

    # Helper to pick a dtype: prefer bfloat16 if available on this torch build, else float16
    dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16

    # Helper: compute conservative per-device max_memory mapping
    def _compute_max_memory_per_device(reserve_gb: int = 4):
        max_mem = {}
        if not torch.cuda.is_available():
            return max_mem
        for i in range(torch.cuda.device_count()):
            total_gb = int(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3))
            usable = max(0, total_gb - reserve_gb)
            max_mem[i] = f"{usable}GB"
        return max_mem

    # Load tokenizer (allow remote code); fall back to slow tokenizer if necessary
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e_fast:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            print("⚠️  Fast tokenizer failed, loaded slow tokenizer (use_fast=False).")
        except Exception as e_slow:
            raise RuntimeError(
                f"Failed to load tokenizer for {model_name}. First error: {e_fast}\nFallback error: {e_slow}\n"
                "Try updating transformers/tokenizers or check the model repo for a custom tokenizer."
            ) from e_slow

    # Prepare to load model with multi-GPU + offload-friendly options.
    max_memory = _compute_max_memory_per_device(reserve_gb=4)

    # clear caches before heavy ops
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    model = None
    attempts = []

    # 1) Try 4-bit quantization (nf4) with CPU offload hints
    try:
        q4_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=q4_cfg,
            low_cpu_mem_usage=True,
            max_memory=max_memory if max_memory else None,
        )
        print("✅ EXAONE loaded with 4-bit quantization + CPU offload (preferred)")
    except Exception as e_q4:
        attempts.append(("4bit", str(e_q4)))
        print("⚠️ 4-bit load failed:", e_q4)

    # 2) Fallback to 8-bit quantization
    if model is None:
        try:
            q8_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                quantization_config=q8_cfg,
                low_cpu_mem_usage=True,
                max_memory=max_memory if max_memory else None,
            )
            print("✅ EXAONE loaded with 8-bit quantization + CPU offload")
        except Exception as e_q8:
            attempts.append(("8bit", str(e_q8)))
            print("⚠️ 8-bit load failed:", e_q8)

    # 3) Final fallback: try normal load (may OOM)
    if model is None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory=max_memory if max_memory else None,
            )
            print("✅ EXAONE loaded in full-precision (no quantization)")
        except Exception as e_full:
            attempts.append(("full", str(e_full)))
            raise RuntimeError(
                f"Failed to load model {model_name}. Attempts: {attempts}\n"
                "Consider using more GPUs, larger GPU memory, or hosted inference."
            ) from e_full

    # If reasoning is requested, user should call tokenizer.apply_chat_template with enable_thinking=True
    # when preparing inputs. We return the raw model and tokenizer so caller can prepare inputs as needed.
    # Wrap the (model, tokenizer) pair in a small callable wrapper compatible with the rest of the
    # codebase which expects an object with an `invoke(prompt)` method (like HuggingFacePipeline).
    class EXAONEWrapper:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

        def invoke(self, prompt: str, reasoning: bool = False, max_new_tokens: int = 128, **kwargs) -> str:
            """Generate text from a single user prompt.

            Args:
                prompt: user prompt string
                reasoning: whether to enable reasoning (<think> block)
                max_new_tokens: tokens to generate
                kwargs: additional generation kwargs (temperature, top_p, do_sample, presence_penalty)

            Returns:
                Decoded generation string
            """
            messages = [{"role": "user", "content": prompt}]

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=reasoning,
            )

            # Determine device: try to infer from model parameters
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                # fallback to cpu
                device = torch.device("cpu")

            # Default generation params recommended for EXAONE
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
            }
            if reasoning:
                gen_kwargs.update({"do_sample": True, "temperature": 0.6, "top_p": 0.95})
            else:
                gen_kwargs.update({"do_sample": False, "temperature": 0.3})

            # Merge any user-provided overrides
            gen_kwargs.update(kwargs)

            # model.generate may not accept presence_penalty directly; warn if provided
            if "presence_penalty" in gen_kwargs:
                # presence_penalty requires logits processing or a higher-level wrapper; remove it here
                gen_kwargs.pop("presence_penalty")
                print("⚠️ presence_penalty requested but not applied directly; implement custom logits warping if needed.")

            input_ids = input_ids.to(device)

            output = self.model.generate(input_ids, **gen_kwargs)

            return self.tokenizer.decode(output[0], skip_special_tokens=True)

    return EXAONEWrapper(model, tokenizer)