#!/bin/bash
#SBATCH -J summary
#SBATCH -p batch
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p hgx
#SBATCH -o inf.out


WORK_DIR=/cognitive_comp/fanyuchen/workspace/GTS-Engine
PRETRAINED_DIR=/cognitive_comp/fanyuchen/workspace/GTS-Engine/pretrained
TASK_DIR=/cognitive_comp/fanyuchen/workspace/GTS-Engine/tasks/text_summary_example

python -u /cognitive_comp/fanyuchen/workspace/GTS-Engine/gts_engine/gts_engine_inference.py \
    --task_dir=$TASK_DIR \
    --engine_type=bagualu \
    --task_type=summary \
    --input_path=examples/summary/lcsts_test.json \
    --output_path=$TASK_DIR/output.json \
