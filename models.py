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

def load_solar():
    """Load Solar model (Llama-2-70b-instruct-v2) from Upstage"""
    tokenizer = AutoTokenizer.from_pretrained("upstage/Llama-2-70b-instruct-v2")
    model = AutoModelForCausalLM.from_pretrained(
        "upstage/Llama-2-70b-instruct-v2",
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        rope_scaling={"type": "dynamic", "factor": 2}  # allows handling of longer inputs
    )
    
    # Create a pipeline for use with LangChain
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=512, 
        truncation=True,
        torch_dtype=torch.float16
    )
    return HuggingFacePipeline(pipeline=pipe)

def load_solar_pro():
    """Load Solar Pro model (solar-pro-preview-instruct) from Upstage"""
    tokenizer = AutoTokenizer.from_pretrained("upstage/solar-pro-preview-instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "upstage/solar-pro-preview-instruct",
        device_map="cuda",  
        torch_dtype="auto",  
        trust_remote_code=True,
    )
    
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