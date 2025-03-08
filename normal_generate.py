import torch
import json
import math
from prompt import get_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate math Q&A data using a pre-trained model.")
    parser.add_argument("--prompt_name", type=str, required=True, help="Prompt name")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    return parser.parse_args()

def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise Exception(f"Failed to load data from {file_path}: {str(e)}")

def format_input(system, data, tokenizer):
    global prompt
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt.format(data=data)}
    ]
    return tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

def main():
    args = parse_args()
    torch.backends.cudnn.enable = True
    device_id = 2
    token = None
    model_mapping = {
        "llama3.1-8b-ins": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3-8b-ins": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama3-8b": "meta-llama/Meta-Llama-3-8B"
    }
    
    if args.model_name not in model_mapping:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    model_name = model_mapping[args.model_name]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16
        ).to(f"cuda:{device_id}")
    except Exception as e:
        raise Exception(f"Failed to load model or tokenizer: {str(e)}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    global prompt
    prompt = get_prompt(args.prompt_name)
    system = "You are an expert at solving mathematical problems."
    
    data_path = "/home/test_yanjunchi/wangshaobo/ICL_Synthesis/data/gsm8k_train.json"
    data = load_data(data_path)
    
    formatted_inputs = [format_input(system, data[i*4:i*4+4], tokenizer) 
                       for i in range(len(data) // 4)]
    
    batch_size = 32
    responses = []
    
    for i in tqdm(range(math.ceil(len(formatted_inputs) / batch_size)), 
                 desc="Processing batches", unit="batch"):
        batch_inputs = formatted_inputs[i * batch_size:(i + 1) * batch_size]
        
        model_inputs = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(f'cuda:{device_id}')
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=1024,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True
            )
        
        generated_ids = generated_ids.sequences
        new_token_ids = [
            generated[model_inputs.input_ids.shape[1]:] 
            for generated in generated_ids
        ]
        
        batch_responses = [
            tokenizer.decode(new_tokens, skip_special_tokens=True)
            for new_tokens in new_token_ids
        ]
        responses.extend(batch_responses)
        
        del model_inputs

    output_data = [
        {
            "instruction": system,
            "input": prompt.format(data=data[i*4:i*4+4]),
            "output": responses[i]
        }
        for i in range(len(data) // 4)
    ]
    
    info = {
        "model_name": args.model_name,
        "prompt": prompt,
        "source_data": "gsm8k_train"
    }
    
    try:
        with open(f"./llama3_gsm8k_4shot_{args.prompt_name}.json", "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        with open(f"./llama3_gsm8k_4shot_{args.prompt_name}_info.json", "w", encoding='utf-8') as f:
            json.dump(info, f, indent=4)
    except Exception as e:
        raise Exception(f"Failed to save output files: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)