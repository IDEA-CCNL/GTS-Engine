#!/bin/bash

set -x

WORK_DIR=$(dirname $(dirname $(dirname $(readlink -f "$0"))))
echo "working directory: $WORK_DIR"

cd $WORK_DIR
mkdir -p $WORK_DIR/tasks
mkdir -p $WORK_DIR/pretrained

PRETRAINED_DIR=$WORK_DIR/pretrained
TASK_DIR=$WORK_DIR/tasks/text_ie_example
mkdir -p $TASK_DIR

export CUDA_VISIBLE_DEVICES=4
python gts_engine/gts_engine_train.py \
    --engine_type=bagualu \
    --train_mode=standard \
    --task_dir=$TASK_DIR \
    --task_type=ie \
    --train_data=weibo_train.json \
    --valid_data=weibo_val.json \
    --test_data=weibo_test.json \
    --data_dir=$WORK_DIR/examples/ner \
    --save_path=$TASK_DIR/outputs \
    --pretrained_model_dir=$PRETRAINED_DIR \
    --train_batchsize=2 \
    --valid_batchsize=4 \
    --max_len=512 \
    --max_epochs=1 \
    --min_epochs=1 \
    --seed=123
