#!/bin/bash

WORK_DIR=$(dirname $(dirname $(dirname $(readlink -f "$0"))))
echo "working directory: $WORK_DIR"

cd $WORK_DIR
mkdir -p $WORK_DIR/tasks
mkdir -p $WORK_DIR/pretrained

PRETRAINED_DIR=$WORK_DIR/pretrained
TASK_DIR=$WORK_DIR/tasks/text_generation_example
mkdir -p $TASK_DIR

export CUDA_VISIBLE_DEVICES=1
python gts_engine/gts_engine_train.py \
    --engine_type=qiankunding \
    --train_mode=standard \
    --task_dir=$TASK_DIR \
    --task_type=generation \
    --train_data=kpg_train.json \
    --valid_data=kpg_val.json \
    --test_data=kpg_test.json \
    --data_dir=$WORK_DIR/examples/text_generation \
    --save_path=$TASK_DIR/outputs \
    --pretrained_model_dir=$PRETRAINED_DIR \
    --train_batchsize=32 \
    --valid_batchsize=32 \
    --max_len=256 \
    --max_epochs=2 \
    --min_epochs=2 \
    --seed=123