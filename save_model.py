import os
import torch
from argparse import ArgumentParser
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


parser = ArgumentParser()
parser.add_argument('--base_model', type=str, help='Directory or repo of the base model')
parser.add_argument('--new_model', type=str, help='Directory of the new model')
parser.add_argument('--save_dir', type=str, help='Directory to save the new model merged with the base model')
parser.add_argument('--push_to_hub', type=bool, help='Push the new model to huggingface model hub', default=False)


def save_model(base_model_name, new_model, save_dir, push_to_hub=False):
    # model_name = "sinarashidi/llama-2-7b-chat-persian"
    # new_model = "llama-2-sentiment-claim-stance"
    device_map = {"": 0}

    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    if push_to_hub:
        model.push_to_hub(new_model, use_temp_dir=False)
        tokenizer.push_to_hub(new_model, use_temp_dir=False)


if __name__ == "__main__":
    args = parser.parse_args()
    base_model_name = args.base_model
    new_model = args.new_model
    save_dir = args.save_dir
    push_to_hub = args.push_to_hub

    save_model(
        base_model_name=base_model_name,
        new_model=new_model,
        save_dir=save_dir,
        push_to_hub=push_to_hub
               )
    