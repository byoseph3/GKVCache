from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch
import time
import math
import psutil

# Model Name
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # or "meta-llama/Llama-2-7b-chat-hf"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)

# Load inference model
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Look for GPU, move infernece model to GPU if avaliable otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def bench(prompt, max_length=50, runs=5, isCached=False, isDistributed=False):
    # '.to' to the device tensors (n-arrays) to put on.
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warm up
    _ = model.generate(**inputs, max_length=max_length)

    times = []
    vram_usages = []
    cpu_usages = []
    thisProc = psutil.Process()
    thisProc.cpu_percent()

    for _ in range(runs):
        start_time = time.time()
        with torch.no_grad(): # No gradient
            output_ids = model.generate(**inputs, max_length=max_length)
        times.append(time.time() - start_time)
        vram_usages.append(psutil.virtual_memory().percent)
        cpu_usages.append(thisProc.cpu_percent())

        # decode
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # core latency metrics
    mid = math.floor(runs / 2)
    avg_time = sum(times) / runs
    first_token_time = times[0]
    rest_tokens_time = sum(times) - first_token_time
    times.sort()
    worst_time = times[0]
    best_time = times[runs-1]
    median_time = times[mid]
    print("==========================")
    print(f"Average Inference Time: {avg_time:.4f} sec")
    print(f"Best Inference Time: {best_time:.4f} sec")
    print(f"Median Inference Time: {median_time:.4f} sec")
    print(f"Worst Inference Time: {worst_time:.4f} sec")
    print(f"First Token Generation Time: {first_token_time:.4f} sec")
    print(f"Second Token to Last Token Generation Time: {rest_tokens_time:.4f} sec")

    # vram
    print("==========================")
    avg_vram_usage = sum(vram_usages) / runs
    vram_usages.sort()
    best_vram_usage = vram_usages[0]
    median_vram_usage = vram_usages[mid]
    worst_vram_usage = vram_usages[runs-1]
    print(f"Average VRAM Usage: {avg_vram_usage:.4f} % VRAM Usage")
    print(f"Best VRAM Usage: {best_vram_usage:.4f} % VRAM Usage")
    print(f"Median VRAM Usage: {median_vram_usage:.4f} % VRAM Usage")
    print(f"Worst VRAM Usage: {worst_vram_usage:.4f} % VRAM Usage")

    # cpu usage
    print("==========================")
    avg_cpu_usage = sum(cpu_usages) / runs
    cpu_usages.sort()
    best_cpu_usage = cpu_usages[0]
    median_cpu_usage = cpu_usages[mid]
    worst_cpu_usage = cpu_usages[runs-1]
    print(f"Average CPU Usage: {avg_cpu_usage:.4f} % CPU Usage")
    print(f"Best CPU Usage: {best_cpu_usage:.4f} % CPU Usage")
    print(f"Median CPU Usage: {median_cpu_usage:.4f} % CPU Usage")
    print(f"Worst CPU Usage: {worst_cpu_usage:.4f} % CPU Usage")

    # tokens per second
    num_tokens = max_length
    throughput = num_tokens/avg_time
    print(f"Tokens per second: {throughput:.2f}")

    # Quality Metrics (ie correctness)

    if isDistributed:
        # Requests per Second
        # Disk I/O
        # Network I/O
        pass

    if isCached:
        # Cache-Only Metrics
        # Hit Rate
        # Miss Rate
        # Eviction Rate
        # Cache Lookup Time
        pass

    # Resource Metrics
    # GPU
    # CPU
    # VRAM
    # RAM
    # Energy Consumption (?)

    # memory or vram usage
    if torch.cuda.is_available():
        print(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    

    print(response)

bench("What is hugging face?")
