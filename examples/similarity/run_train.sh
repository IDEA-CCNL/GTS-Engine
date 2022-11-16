#!/bin/bash

WORK_DIR=$(dirname $(dirname $(dirname $(readlink -f "$0"))))
echo "working directory: $WORK_DIR"

cd $WORK_DIR
mkdir -p $WORK_DIR/tasks
mkdir -p $WORK_DIR/pretrained

PRETRAINED_DIR=$WORK_DIR/pretrained
TASK_DIR=$WORK_DIR/tasks/similarity_example
mkdir -p $TASK_DIR

export CUDA_VISIBLE_DEVICES=0
python gts_engine/gts_engine_train.py \
    --task_dir=$TASK_DIR \
    --task_type=similarity \
    --train_data=train.json \
    --valid_data=dev.json \
    --test_data=test.json \
    --data_dir=$WORK_DIR/examples/similarity \
    --save_path=$TASK_DIR/outputs \
    --pretrained_model_dir=$PRETRAINED_DIR \
    --train_batchsize=2 \
    --valid_batchsize=4 \
    --max_len=512 \
    --max_epochs=1 \
    --min_epochs=1 \
    --seed=123