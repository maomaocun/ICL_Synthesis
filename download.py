import subprocess
from multiprocessing import Pool

def download_model(command):
    subprocess.run(command, shell=True)

commands = [
    "modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct --local_dir ./model/Meta-Llama-3.1-8B-Instruct",
    "modelscope download --model LLM-Research/Meta-Llama-3-8B-Instruct --local_dir ./model/Meta-Llama-3-8B-Instruct",
    "modelscope download --model LLM-Research/Meta-Llama-3-8B --local_dir ./model/Meta-Llama-3-8B"
]

if __name__ == "__main__":
    with Pool(processes=3) as pool:
        pool.map(download_model, commands)