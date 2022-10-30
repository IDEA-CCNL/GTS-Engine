CUDA_VISIBLE_DEVICES=0 
python train.py \
    --train_data=train.json \
    --valid_data=dev.json \
    --test_data=test_label.json \
    --labels_data=labels.json \
    --data_dir=files/data \
    --save_path=files \
    --train_batchsize=2 \
    --valid_batchsize=4 \
    --max_len=512 \
    --task_id=100 \
    --tuning_method=UnifiedMC \
    --max_epochs=1 \
    --min_epochs=1 \
    --num_threads=8 \
    --seed=123 \
    --val_check_interval=0.25 > train_100.log &