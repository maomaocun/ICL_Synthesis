import argparse
import threading
import torch
import json
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 定义处理函数
def process_data(data_slice, model_name, device, thread_id, results, lock, batch_size=64):
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=f"cuda:{device}",
        torch_dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # 格式化输入
    def format_input(system, prompt):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    system = "You are an expert at solving mathematical problems. Please think step by step! Please note that the final answer should be in the format of ###{answer} and at the end of the reply."
    formatted_inputs = [format_input(system, item["question"]) for item in data_slice]

    responses = []
    for i in tqdm(range(math.ceil(len(formatted_inputs) / batch_size)), desc=f"Thread-{thread_id} processing", unit="batch"):
        batch_inputs = formatted_inputs[i * batch_size:(i + 1) * batch_size]
        model_inputs = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(f"cuda:{device}")

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )

        generated_ids = generated_ids.sequences
        new_token_ids = [
            generated[model_inputs.input_ids.shape[1]:]  # 只取生成部分 (跳过输入部分)
            for generated in generated_ids
        ]
        batch_responses = [
            tokenizer.decode(new_tokens, skip_special_tokens=True)
            for new_tokens in new_token_ids
        ]
        responses.extend(batch_responses)

    # 保存结果到共享列表
    with lock:
        for i in range(len(data_slice)):
            results.append({
                "instruction": system,
                "input": data_slice[i]["question"],
                "output": responses[i],
            })


# 主函数
def main(args):
    with open(args.input_file, "r") as f:
        data = json.load(f)

    # 分割数据
    num_gpus = len(args.gpus)
    data_slices = [data[i::num_gpus] for i in range(num_gpus)]

    threads = []
    results = []
    lock = threading.Lock()

    for thread_id, (device, data_slice) in enumerate(zip(args.gpus, data_slices)):
        thread = threading.Thread(
            target=process_data,
            args=(data_slice, args.model_name, device, thread_id, results, lock, args.batch_size),
        )
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    # 保存最终结果
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU LLM Data Processing")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--gpus", type=int, nargs="+", required=True, help="List of GPU IDs to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing")

    args = parser.parse_args()
    main(args)
