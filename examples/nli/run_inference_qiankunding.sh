#!/bin/bash

WORK_DIR=$(dirname $(dirname $(dirname $(readlink -f "$0"))))
echo "working directory: $WORK_DIR"

cd $WORK_DIR

TASK_DIR=$WORK_DIR/tasks/nli_example

if [ ! -d $TASK_DIR ]; then
    echo "task dir $TASK_DIR not exists, please train first."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
python gts_engine/gts_engine_inference.py \
    --task_dir=$TASK_DIR \
    --engine_type=qiankunding \
    --task_type=nli \
    --input_path=examples/nli/test.json \
    --output_path=$TASK_DIR/output.json
