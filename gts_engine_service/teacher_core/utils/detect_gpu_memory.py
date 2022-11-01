from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def detect_gpu_memory():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    # print(info)

    return info.total//1024**2, info.used//1024**2

def decide_gpu(gpu_memory):
    low_gpu = 25000
    mid_gpu = 33000
    high_gpu = 41000

    if gpu_memory <= low_gpu:
        return "low_gpu"
    elif gpu_memory > low_gpu and gpu_memory < mid_gpu:
        return "mid_gpu"
    elif gpu_memory >= mid_gpu:
        return "high_gpu"

if __name__=="__main__":
    print_gpu_utilization()
    gpu_memory, gpu_used_memory = detect_gpu_memory()
    print(gpu_memory, gpu_used_memory)
