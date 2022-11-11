# GTS-Engine

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
</p>


<h4 align="center">
  <a href=#安装> 安装 </a> |
  <a href=#快速开始> 快速开始 </a> |
  <a href=#API文档> API文档 </a>
</h4>

------------------------------------------------------------------------------------------

GTS-Engine是一款开箱即用且性能强大的自然语言理解引擎，能够仅用小样本就能自动化生产NLP模型。它依托于封神榜开源体系的基础模型，并在下游进行了有监督预训练，同时集成了多种小样本学习技术，搭建了一个模型自动生产的流水线。

GTS-Engine计划开源两个系列的引擎，分别为**乾坤鼎**系列和**八卦炉**系列。
- 乾坤鼎系列是以1.3B参数的大模型为底座，通过大模型进行训练和推理的引擎。
- 八卦炉系列是以110M参数的base级别模型为底座，融合大模型、数据增强、协调训练等方法进行训练和推理的引擎。

本次发布的是**乾坤鼎**系列的Beta版本，更多的功能更新请持续关注我们的Github。

## 安装

#### 环境需求和软件依赖

- 软件环境依赖
    - Python >= 3.7
    - 其他依赖请参考`requirements.txt`
- 硬件环境需求
    - 乾坤鼎引擎至少需要一张24G显存的RTX3090，使用V100和A100能够获得更好的性能体验；

更多环境需求和软件依赖请参考我们的文档。

#### pip安装

您可以通过pip直接进行安装。

```bash
pip install gts-engine
```

#### 直接安装

也可以clone下github项目后进行安装。

```bash
git clone https://github.com/IDEA-CCNL/GTS-Engine.git
cd GTS-Engine
python setup.py install
```

#### 使用Docker

我们提供一个打包好GTS-Engine的Docker来运行我们的引擎。

```bash
docker run -it --name  gst_engine \
-p 5201:5201 \
--mount type=bind,source=/raid/liuyibo/GTS-Engine/files,target=/workspace/gts_teacher_service/files \
gts_engine_image:v0  

#运行api.py
CUDA_VISIBLE_DEVICES=0 python api.py
```

#### Python SDK

建议您通过我们编写的Python SDK来使用GTS-Engine的服务，请参考[GTS-Engine-Client](https://github.com/IDEA-CCNL/GTS-Engine-Client)。

## 快速开始

#### 启动服务

- 您可以直接通过调用命令行启动GTS-Engine的服务。

```bash
git clone https://github.com/IDEA-CCNL/GTS-Engine.git #下载源码
mkdir pretrained  #将下载好的模型文件放在pretrained
cd GTS-Engine/gts_engine_service
CUDA_VISIBLE_DEVICES=0 python api.py #指定GPU 运行api.py
```

- 同时也可以通过Docker命令启动Docker来运行我们的服务。

```bash
#启动docker
docker run -it --name  gst_engine \
-p 5201:5201 \
--mount type=bind,source=/raid/liuyibo/GTS-Engine/files,target=/workspace/gts_teacher_service/files \
gts_engine_image:v0  

#运行api.py
CUDA_VISIBLE_DEVICES=0 python api.py
```

#### 开始训练

结合GTS-Engine-Client，您可以仅通过八行代码即可完成模型的训练。

```python
from gts_engine_client import GTSEngineClient
#ip和port参数与启动服务的ip和port一致
client = GTSEngineClient(ip="192.168.190.2", port="5207")
# 创建任务
client.create_task(task_name="tnews_classification", task_type="classification")
# 上传文件
client.upload_file(task_id="tnews_classification", local_data_path="train.json")
client.upload_file(task_id="tnews_classification", local_data_path="dev.json")
client.upload_file(task_id="tnews_classification", local_data_path="test.json")
client.upload_file(task_id="tnews_classification", local_data_path="labels.json")
# 开始训练
client.start_train(task_id="tnews_classification", train_data="train.json", val_data="dev.json", test_data="test.json", label_data="labels.json", gpuid=0)
```

#### 开始推理

同样地，您也可以在训练完成后，仅使用三行代码完成推理。

```python
from gts_engine_client import GTSEngineClient
# 加载已训练好的模型
client.start_inference(task_id="tnews_classification")
# 预测
client.inference(task_id="tnews_classification", samples=[{"content":"怎样的房子才算户型方正？"}, {"content":"文登区这些公路及危桥将进入 封闭施工，请注意绕行！"}])
```

更多快速使用的详情，请参考我们的文档。

## API文档

更多GTS-Engine的内容可参考API文档。

## 效果展示

GTS-Engine将专注于解决各种自然语言理解任务。乾坤鼎引擎通过一套训练流水线，已经达到了人类算法专家的水准。2022年11月11日，GTS乾坤鼎引擎在中文语言理解权威评测基准FewCLUE榜单上登顶。GTS-Engine系列会持续在各个NLU任务上不断优化，持续集成，带来更好的开箱即用的体验。

## 即将发布

- 分类任务增加高级模式，支持用户上传无标注数据进行Self Training，进一步提升效果；
- 更好的使用体验，代码快速迭代中；
- 增加信息抽取任务，SOTA效果即将公开；

## 引用

如果您在研究中使用了我们的工具，请引用我们的工作：

```
@misc{GTS-Engine,
  title={GTS-Engine},
  author={IDEA-CCNL},
  year={2022},
  howpublished={\url{https://github.com/IDEA-CCNL/GTS-Engine}},
}
```

## 开源协议

GTS-Engine遵循[Apache-2.0开源协议](https://github.com/IDEA-CCNL/GTS-Engine/blob/main/LICENSE)。