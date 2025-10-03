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