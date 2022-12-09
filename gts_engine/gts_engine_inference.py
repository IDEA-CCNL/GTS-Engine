import os
import sys
import json
import argparse

# 如果没有安装gts_engine，请把GTS-Engine/gts-engine加入到系统环境变量
sys.path.append(os.path.dirname(__file__))

from gts_common.registry import PIPELINE_REGISTRY
from pipelines import *
# 设置gpu相关的全局变量
import qiankunding.utils.globalvar as globalvar
globalvar._init()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------推理模型的接口，按照任务类型调用------------------------------------------
def preprare_inference(engine_type, task_type, save_path):
    inference_module = "pipelines." + engine_type + "_" + task_type
    prepare_inference_func = PIPELINE_REGISTRY.get(name="prepare_inference", suffix=inference_module)
    return prepare_inference_func(save_path)

def inference_samples(engine_type, task_type, samples, inference_suite):
    inference_module = "pipelines." + engine_type + "_" + task_type
    inference_func = PIPELINE_REGISTRY.get(name="inference", suffix=inference_module)
    return inference_func(samples, inference_suite)

def main():
    total_parser = argparse.ArgumentParser()

    total_parser.add_argument("--task_dir", required=True, 
                            type=str, help="specific task directory")
    total_parser.add_argument("--engine_type", required=True, choices=["qiankunding", "bagualu"],
                            type=str, help="engine type")
    total_parser.add_argument("--task_type", required=True, choices=["classification", "similarity", "nli", "ie"],
                            type=str, help="task type for training")
    total_parser.add_argument("--input_path", required=True,
                            type=str, help="input path of data which will be inferenced")
    total_parser.add_argument("--output_path", required=True,
                            type=str, help="output path of inferenced data")
    
    args = total_parser.parse_args()                            

    save_path = os.path.join(args.task_dir, "outputs")
    samples = []
    for line in open(args.input_path):
        line = line.strip()
        sample = json.loads(line)
        samples.append(sample)
    
    inference_suite = preprare_inference(args.engine_type, args.task_type, save_path)
    result = inference_samples(args.engine_type, args.task_type, samples, inference_suite)

    with open(args.output_path, encoding="utf8", mode="w") as fout:
        json.dump(result, fout, ensure_ascii=False)

if __name__ == '__main__':    
    main()
    