#!/bin/bash
#SBATCH -J summary
#SBATCH -p batch
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p hgx
#SBATCH -o summary.out


WORK_DIR=/cognitive_comp/fanyuchen/workspace/GTS-Engine
PRETRAINED_DIR=/cognitive_comp/fanyuchen/workspace/GTS-Engine/pretrained
TASK_DIR=/cognitive_comp/fanyuchen/workspace/GTS-Engine/tasks/text_summary_example

python -u /cognitive_comp/fanyuchen/workspace/GTS-Engine/gts_engine/gts_engine_train.py \
    --engine_type=bagualu \
    --train_mode=standard \
    --task_dir=$TASK_DIR \
    --task_type=summary \
    --train_data=lcsts_train.json \
    --valid_data=lcsts_val.json \
    --test_data=lcsts_test.json \
    --data_dir=$WORK_DIR/examples/summary \
    --save_path=$TASK_DIR/outputs \
    --pretrained_model_dir=$PRETRAINED_DIR \
    --train_batchsize=8 \
    --valid_batchsize=4 \
    --max_len=512 \
    --max_epochs=6 \
    --min_epochs=1 \
    --seed=123
