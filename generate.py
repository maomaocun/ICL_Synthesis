from vllm import LLM
import json
import time
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer
import torch.distributed as dist
import os
import torch.multiprocessing as mp

# 将多进程启动方式设置为 spawn
mp.set_start_method('spawn', force=True)
# def get_available_gpus():
#     available_gpus = []
    
#     for i in range(torch.cuda.device_count()):
#         if torch.cuda.is_available():
#             available_gpus.append(str(i))
    
#     return available_gpus
import torch
import subprocess

def get_available_gpus(max_memory_usage_gb=2):
    """
    获取所有显存占用小于 max_memory_usage_gb 的 GPU。
    
    :param max_memory_usage_gb: 最大允许的显存使用量（单位：GB）。
    :return: 可用 GPU 的索引列表。
    """
    available_gpus = []
    
    try:
        # 获取 GPU 显存使用信息
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # 解析显存使用信息
        memory_usage = result.stdout.strip().split("\n")
        memory_usage = [int(mem) / 1024 for mem in memory_usage]  # 转换为 GB
        
        # 筛选显存占用小于 max_memory_usage_gb 的 GPU
        for i, mem in enumerate(memory_usage):
            if mem < max_memory_usage_gb:
                available_gpus.append(str(i))
    except FileNotFoundError:
        print("nvidia-smi 未找到，请确保 NVIDIA 驱动已正确安装并加入 PATH 环境变量。")
    except subprocess.CalledProcessError as e:
        print(f"获取 GPU 信息时出错: {e}")
    
    return available_gpus

# 获取所有可用的 GPU
available_gpus = get_available_gpus()
available_gpus = available_gpus[:1]
if len(available_gpus) == 3:
    available_gpus = available_gpus[:2]
if available_gpus:
    # 设置 CUDA_VISIBLE_DEVICES 为所有可用 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(available_gpus)
    print(f"Using the following GPUs: {', '.join(available_gpus)}")
else:
    print("No available GPUs.")

# sampling_params = SamplingParams(
#     temperature=0,
#     max_tokens=512,
#     top_p=0.9,
#     top_k=50,
#     repetition_penalty=1.3,  # 惩罚重复
#     )
sampling_params = SamplingParams(
    # temperature=0.5, 
    # top_p=0.95,
    max_tokens=512,
    # repetition_penalty=1.2,
    # early_stopping = True,
    stop_token_ids=[128009],
    include_stop_str_in_output=False
)
# sampling_params = SamplingParams(
#     n=1,
#     best_of=1,
#     presence_penalty=0.6,
#     frequency_penalty=0.5,
#     repetition_penalty=1.3,
#     temperature=0.7,
#     top_p=0.9,
#     top_k=50,
#     min_p=0.1,
#     seed=42,
#     # use_beam_search=True,
#     # length_penalty=1.0,
#     # early_stopping=True,
#     stop=["END"],
#     stop_token_ids=[50256],
#     include_stop_str_in_output=False,
#     # ignore_eos=True,
#     max_tokens=256,
#     # logprobs=5,
#     # prompt_logprobs=3,
#     skip_special_tokens=True,
#     spaces_between_special_tokens=True,
#     logits_processors=[]
# )
# device_id = 0
torch.cuda.empty_cache()
system = "You are an expert at solving mathematical problems. Please think step by step!"
def format_input(system, prompt):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt
model_name_or_path = '/home/test_yanjunchi/wangshaobo/ICL_Synthesis/model1/LLM-Research/Meta-Llama-3___1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

llm = LLM(
    model=model_name_or_path,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=len(available_gpus), 
    device=f"cuda",
    dtype=torch.bfloat16,
    max_model_len=8192  
)

with open("./data/train.json","r") as f:
    data = json.load(f)
    
# data = data[:10]
data_input = []
# data = [{"question":"What's your name?"}]
for i in data:
    data_input.append(format_input(system, i["question"]))
# prompts = []
# for i in data:
#     prompts.append(i["question"])
print(data_input[0])
response = llm.generate(data_input,sampling_params)
ans = []
print(response[0])
for i in response:
    ans.append(i.outputs[0].text)

for index in range(len(ans)):
    ans[index] = {
        "instruction": system,
        "input": data[index]["question"],
        "output": ans[index] 
        "ground_truth": data[index]["answers"]
    }  
with open("llama3.1_gsm8k_base_generate1.json","w") as f:

    json.dump(ans,f,indent=4)
    
if dist.is_initialized():
    dist.destroy_process_group()
    
##################
# prompt = "请帮我写一篇800字的关于“春天”的作文"
# messages = [
#     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# llm = LLM(
#     model="gpt2",
#     tokenizer="gpt2-tokenizer",
#     tokenizer_mode="auto",
#     trust_remote_code=True,
#     tensor_parallel_size=2,
#     dtype="float16",
#     quantization=None,
#     revision="v1.0",
#     seed=42,
#     gpu_memory_utilization=0.8,
#     swap_space=4,
#     enforce_eager=False,
#     max_context_len_to_capture=512,
#     disable_custom_all_reduce=False
# )

# sampling_params = SamplingParams(
#     n=3,
#     best_of=5,
#     presence_penalty=0.5,
#     frequency_penalty=0.5,
#     repetition_penalty=1.2,
#     temperature=0.7,
#     top_p=0.9,
#     top_k=50,
#     min_p=0.1,
#     seed=42,
#     use_beam_search=True,
#     length_penalty=1.0,
#     early_stopping=True,
#     stop=["END"],
#     stop_token_ids=[50256],
#     include_stop_str_in_output=False,
#     ignore_eos=True,
#     max_tokens=50,
#     logprobs=5,
#     prompt_logprobs=3,
#     skip_special_tokens=True,
#     spaces_between_special_tokens=True,
#     logits_processors=[]
# )