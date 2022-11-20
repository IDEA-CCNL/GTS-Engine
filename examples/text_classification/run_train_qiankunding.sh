#!/bin/bash

WORK_DIR=$(dirname $(dirname $(dirname $(readlink -f "$0"))))
echo "working directory: $WORK_DIR"

cd $WORK_DIR
mkdir -p $WORK_DIR/tasks
mkdir -p $WORK_DIR/pretrained

PRETRAINED_DIR=$WORK_DIR/pretrained
TASK_DIR=$WORK_DIR/tasks/text_classification_example
mkdir -p $TASK_DIR

export CUDA_VISIBLE_DEVICES=3 
python gts_engine/gts_engine_train.py \
    --engine_type=qiankunding \
    --task_dir=$TASK_DIR \
    --task_type=classification \
    --train_data=tnews_train.json \
    --valid_data=tnews_val.json \
    --test_data=tnews_test.json \
    --label_data=tnews_label.json \
    --data_dir=$WORK_DIR/examples/text_classification \
    --save_path=$TASK_DIR/outputs \
    --pretrained_model_dir=$PRETRAINED_DIR \
    --train_batchsize=2 \
    --valid_batchsize=4 \
    --max_len=512 \
    --max_epochs=1 \
    --min_epochs=1 \
    --seed=123