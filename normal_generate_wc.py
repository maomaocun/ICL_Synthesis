# import torch
# import json
# import math
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # 加载预训练的分词器和模型
# model_name = '/home/test_yanjunchi/wangshaobo/ICL_Synthesis/model1/LLM-Research/Meta-Llama-3___1-8B-Instruct'
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     device_map="cuda:0", 
#     torch_dtype=torch.bfloat16
# )

# # 设置 pad_token_id，避免与 eos_token_id 冲突
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = 'left'
# # 格式化输入
# def format_input(system, prompt):
#     messages = [
#         {"role": "system", "content": system},
#         {"role": "user", "content": prompt}
#     ]
#     # print(tokenizer.apply_chat_template(
#     #     messages, 
#     #     tokenize=False, 
#     #     add_generation_prompt=False,
#     # ))
#     # print("False")
#     # NOTE , apply_chat_template will add Cutting Knowledge Date and Today Date to system
#     return tokenizer.apply_chat_template(
#         messages, 
#         tokenize=False, 
#         add_generation_prompt=True,
        
#     )

# # 加载数据
# with open("./data/train.json", "r") as f:
#     data = json.load(f)

# # data = data[:10]  # 只处理前 10 条数据
# system = "You are a friendly AI assistance!"
# formatted_inputs = [format_input(system, item["question"]) for item in data]

# # 批量生成回复
# batch_size = 32  # 设置小批量，避免显存不足
# responses = []

# for i in range(math.ceil(len(formatted_inputs) / batch_size)):
#     batch_inputs = formatted_inputs[i * batch_size:(i + 1) * batch_size]
    
#     # Tokenize 输入并生成 attention_mask
#     model_inputs = tokenizer(
#         batch_inputs,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=512
#     ).to('cuda')
    
#     # 生成文本
#     with torch.no_grad():
#         generated_ids = model.generate(
#             input_ids=model_inputs.input_ids,
#             attention_mask=model_inputs.attention_mask,
#             max_new_tokens=512,
#             pad_token_id=tokenizer.pad_token_id,
#             return_dict_in_generate=True,  # 确保返回字典
#         )
    
#     # 解码生成的文本
#     generated_ids = generated_ids.sequences
#     new_token_ids = [
#         generated[model_inputs.input_ids.shape[1]:]  # 只取生成部分 (跳过输入部分)
#         for generated in generated_ids
#     ]

#     # 解码生成的新增部分
#     batch_responses = [
#         tokenizer.decode(new_tokens, skip_special_tokens=True)
#         for new_tokens in new_token_ids
#     ]
#     responses.extend(batch_responses)
#     # batch_responses = tokenizer.batch_decode(
#     #     generated_ids,
#     #     skip_special_tokens=True
#     # )
#     responses.extend(batch_responses)

# # 打印生成的响应
# # for response in responses:
# #     print("-------"*5)
# #     print(response)
# d = []
# for i in range(len(data)):
#     d.append(
#         {
#         "instruction": system,
#         "input":data[i]["question"],
#         "output":responses[i],
#         }
#     )
# with open("llama3.1_gsm8k_base_generate.json", "w") as f:
#     json.dump(d,f,indent=4)
import torch
import json
import math
from prompt import get_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm  # 引入 tqdm 进度条
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Generate math Q&A data using a pre-trained model.")
    parser.add_argument("--prompt_name", type=str,help="prompt name")
    parser.add_argument("--model_name", type=str,help="model name")
    return parser.parse_args()
def read_jsonl(file_path):
    result = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                result.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")
    return result
args = parse_args()
torch.backends.cudnn.enable =True
device_id = 2
# 加载预训练的分词器和模型
if args.model_name == "llama3.1-8b-ins":
    model_name = 'Meta-Llama-3___1-8B-Instruct'
elif args.model_name == "llama3-8b-ins":
    model_name = "Meta-Llama-3-8B-Instruct"
elif args.model_name == "llama3-8b":
    model_name = "Meta-Llama-3-8B"



tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False,token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    # device_map=f"cuda:{device_id}", 
    torch_dtype=torch.bfloat16
).to(f"cuda:{device_id}")

# 设置 pad_token_id，避免与 eos_token_id 冲突
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'


prompt = get_prompt(args.prompt_name)
# 格式化输入
def format_input(system, data):
    global prompt
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt.format(data=data)}
    ]
    return tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
    )

# 加载数据
# with open("../data/gsm8k_train.json", "r") as f:
#     data = json.load(f)
# path1 = "./llama3.1_gsm8k_normal_generate_parse.json"
# path2 = "gsm8k_rand10p.jsonl"
# with open(path1,"r") as f:
#     data1 = json.load(f)    
# data2 = read_jsonl(path2)
# data = data1 + data2
path = "/home/test_yanjunchi/wangshaobo/ICL_Synthesis/data/gsm8k_train.json"
with open(path,"r") as f:
    data = json.load(f)
# data = data[:32]
# data = data[:32]
# for d in data2:
#     k.append({
#         "question":d["input"],
#         "answer":d["output"]
#     })
# data = data[:32]
system = "You are an expert at solving mathematical problems. "
formatted_inputs = [format_input(system, data[i*4:i*4+4]) for i in range(len(data)//4)]
# formatted_inputs = [format_input(system, item) for item in data]

# 批量生成回复
batch_size = 32  # 设置小批量，避免显存不足
responses = []

# 使用 tqdm 显示进度条
for i in tqdm(range(math.ceil(len(formatted_inputs) / batch_size)), desc="Processing batches", unit="batch"):
    batch_inputs = formatted_inputs[i * batch_size:(i + 1) * batch_size]
    
    # Tokenize 输入并生成 attention_mask
    model_inputs = tokenizer(
        batch_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(f'cuda:{device_id}')
    
    # 生成文本
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=1024,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,  # 确保返回字典
        )
    
    # 解码生成的文本
    generated_ids = generated_ids.sequences
    new_token_ids = [
        generated[model_inputs.input_ids.shape[1]:]  # 只取生成部分 (跳过输入部分)
        for generated in generated_ids
    ]

    # 解码生成的新增部分
    batch_responses = [
        tokenizer.decode(new_tokens, skip_special_tokens=True)
        for new_tokens in new_token_ids
    ]
    responses.extend(batch_responses)
    
    # 手动删除不再使用的变量，释放显存
    del model_inputs
    # torch.cuda.empty_cache()  # 清理显存

# 将响应保存到文件

d = []
for i in range(len(data)//4):
    d.append(
        {
        "instruction": system,
        "input": prompt.format(data=data[i*4:i*4+4]),
        "output": responses[i],
        }
    )
info = {
    "model_name": args.model_name,
    "prompt": prompt,
    "source_data": "gsm8k_train"
}
with open(f"./llama3_gsm8k_4shot_{args.prompt_name}.json", "w") as f:
    json.dump(d, f, indent=4)
with open(f"./llama3_gsm8k_4shot_{args.prompt_name}_info.json", "w") as f:
    json.dump(info, f, indent=4)