from vllm import LLM, SamplingParams
import json

# 读取输入数据
input_file = "../../../WenjieFu/GSM8K/train.jsonl"
output_file = "generate.json"
model_path = "/obs/llama3-8b/model"
batch_size = 64
system_info = "You are an expert at solving mathematical problems. Please think step by step! Please note that the final answer should be in the format of ###{answer} and at the end of the reply."

llm = LLM(model=model_path)

with open(input_file, "r") as f:
    inputs = [json.loads(line)["question"] for line in f]

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
outputs = llm.generate(inputs, sampling_params)

with open(output_file, "w") as f:
    json.dump(outputs, f)
