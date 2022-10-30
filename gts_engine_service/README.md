# GTS-Engine


## 操作文档
hold
## 环境依赖
```
pip install -r requirements.txt
```


#### 注意: 预训练资源要求

- 推荐使用至少1张24G以上显存的GPU进行预训练，如3090，A100。


#### 数据准备

- 数据编码：UTF-8
- 数据格式：训练、验证、测试文本数据放在同个目录下，每个文件中为json格式。



#### 单机单卡

```
CUDA_VISIBLE_DEVICES=0 
python train.py 
    --data_dir files/data \
    --save_path files \
    --train_data train.json \
    --valid_data dev.json \
    --test_data test.json \
    --train_batchsize 2 \
    --valid_batchsize 4 \
    --max_len 512 \
    --max_epochs 10 \
    --min_epochs 5 \
    --seed 123 \
    --val_check_interval 0.25 \ 
    --lr 0.1 \
```


可配置参数包括
- ``data_dir``表示训练数据所在目录，该目录下至少要有``train.json``、``val.json``、``label.json``三个文件，如果需要预测```test.json``也要存在。
- ``save_path``表示模型训练参数和训练日志以及训练结果的保存目录。
- ``train_batchsize``表示训练时每次迭代每张卡上的样本数量。当batch_size=2时，max_len=512时, 运行时单卡约需要24G显存。如果实际GPU显存小于24G或大大多于24G，可适当调小/调大此配置。
- ``max_len`` 表示最大句子长度，超过该长度将被截断。
- ``max_epochs`` 表示训练最大轮数。
- ``min_epochs`` 表示训练最小轮数。
- ``seed`` 表示训练随机种子，不同种子可能会出现结果的波动。
- ``val_check_interval`` 表示校验的频率，check_val_every_n_epoch=1表示为每1个epoch校验一次。
- ``lr`` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。

