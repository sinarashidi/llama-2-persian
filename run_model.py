import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

def run(model_name, prompt):
    # model_name = "sinarashidi/llama-2-7b-chat-persian"
    device_map = {"": 0}
    use_4bit = True
    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"
    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"
    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Fix weird overflow issue with fp16 training
    # Ignore warnings
    logging.set_verbosity(logging.CRITICAL)

    # Run text generation pipeline with our next model
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])