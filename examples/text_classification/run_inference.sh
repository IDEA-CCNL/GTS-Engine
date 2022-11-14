#!/bin/bash

WORK_DIR=$(dirname $(dirname $(dirname $(readlink -f "$0"))))
echo "working directory: $WORK_DIR"

cd $WORK_DIR

TASK_DIR=$WORK_DIR/tasks/text_classification_example

if [ ! -d $TASK_DIR ]; then
    echo "task dir $TASK_DIR not exists, please train first."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=7 
python gts_engine/gts_engine_inference.py \
    --task_dir=$TASK_DIR \
    --task_type=classification \
    --input_path=examples/text_classification/tnews_test.json \
    --output_path=tnews_output.json
