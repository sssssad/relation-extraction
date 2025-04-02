# 中文PCNN关系抽取模型

基于分段卷积神经网络(PCNN)实现的中文关系抽取模型，专注于远程监督学习场景。该模型利用PCNN编码器捕获实体间关系，通过包级注意力机制聚合多个句子的信息，提高远程监督关系抽取的效果。

## 项目特点

- 使用**分段卷积神经网络(PCNN)**作为主要编码器
- 支持**远程监督学习**方法，通过包级注意力机制处理同一实体对的多个句子
- 专为**中文文本**优化设计
- 完整的训练、评估和预测流程
- 提供交互式命令行预测界面

## 目录结构

```
chinese_pcnn_relation_extraction/
├── configs/              # 配置文件目录
│   └── bag_pcnn.json     # PCNN包级关系抽取配置
├── data/                 # 数据集目录
├── models/               # 预训练模型和保存的模型目录
├── processors/           # 数据处理和编码器
│   ├── data_processor.py        # 数据处理器
│   └── chinese_pcnn_encoder.py  # 中文PCNN编码器
├── models/               # 模型定义
│   └── chinese_bag_pcnn_model.py # 中文PCNN关系抽取模型
├── train.py              # 训练脚本
├── predict.py            # 预测脚本
├── run_training.py       # 简化训练启动脚本
└── README.md             # 项目文档
```

## 安装

### 环境要求

- Python 3.6+
- PyTorch 1.6+
- scikit-learn
- tqdm
- numpy

### 安装依赖

```bash
pip install torch==1.6.0
pip install scikit-learn==0.24.1
pip install tqdm
pip install numpy
```

## 数据格式

### 包级数据格式(远程监督)

训练数据和验证数据格式：
```
头实体ID\t尾实体ID\t关系\t句子ID\t文本\t头实体\t尾实体
```

示例：
```
Q1	Q2	创始人	S1	乔布斯于1976年在加利福尼亚创立了苹果公司。	乔布斯	苹果公司
Q1	Q2	创始人	S2	苹果公司由史蒂夫·乔布斯、斯蒂夫·沃兹尼亚克和罗恩·韦恩创立。	乔布斯	苹果公司
```

### 关系标签文件格式

```
关系ID\t关系名称
```

示例：
```
0	创始人
1	总部位于
2	出生地
...
```

## 使用方法

### 训练模型

#### 使用配置文件训练

```bash
python run_training.py \
    --config configs/bag_pcnn.json \
    --train_file data/train.txt \
    --dev_file data/dev.txt \
    --relation_file data/relation.txt \
    --output_dir models/pcnn_bag_model
```

#### 自定义参数训练

```bash
python train.py \
    --bag_mode att \
    --train_file data/train.txt \
    --dev_file data/dev.txt \
    --relation_file data/relation.txt \
    --max_seq_length 128 \
    --hidden_size 230 \
    --word_size 50 \
    --position_size 5 \
    --kernel_size 3 \
    --padding_size 1 \
    --dropout 0.1 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --num_train_epochs 30 \
    --output_dir models/pcnn_bag_model
```

### 预测

#### 批量预测

```bash
python predict.py \
    --model_path models/pcnn_bag_model/checkpoint-epoch-10/model.pt \
    --relation_file data/relation.txt \
    --input_file data/test.txt \
    --output_file data/test_result.txt
```

输入文件格式: `文本\t头实体\t尾实体`

#### 交互式预测

```bash
python predict.py \
    --model_path models/pcnn_bag_model/checkpoint-epoch-10/model.pt \
    --relation_file data/relation.txt \
    --interactive
```

在交互式界面中输入格式：
```
文本\t头实体\t尾实体
```

示例：
```
乔布斯于1976年在加利福尼亚创立了苹果公司。	乔布斯	苹果公司
```

## 模型架构

### 分段卷积神经网络(PCNN)编码器

PCNN编码器是CNN的改进版本，通过考虑实体位置信息，将句子分为三个部分进行卷积和池化操作：
1. 头实体之前的部分
2. 两个实体之间的部分
3. 尾实体之后的部分

相比标准CNN，PCNN能够更好地捕获实体周围的上下文信息，提高关系抽取性能。

### 包级注意力机制

在远程监督场景中，对于同一对实体可能有多个句子提及它们的关系。包级注意力机制通过学习为每个句子分配不同的权重，聚合多个句子的信息，更准确地预测实体间关系。

## 示例

### 训练样例
```bash
python run_training.py \
    --config configs/bag_pcnn.json \
    --train_file data/train.txt \
    --dev_file data/dev.txt \
    --relation_file data/relation.txt \
    --output_dir models/pcnn_bag_model
```

### 预测样例
```bash
python predict.py \
    --model_path models/pcnn_bag_model/checkpoint-epoch-10/model.pt \
    --relation_file data/relation.txt \
    --interactive
```

## 数据准备示例

### 关系标签文件 (relation.txt)
```
0	NA
1	创始人
2	毕业于
3	总部位于
4	出生地
5	配偶
```

### 训练数据示例 (train.txt)
```
P1	C1	创始人	S1	乔布斯和沃兹尼亚克在1976年创办了苹果公司。	乔布斯	苹果公司
P1	C1	创始人	S2	史蒂夫·乔布斯是苹果公司的联合创始人。	乔布斯	苹果公司
P2	C2	毕业于	S3	马云毕业于杭州师范学院。	马云	杭州师范学院
P3	C3	总部位于	S4	腾讯公司总部设在深圳。	腾讯公司	深圳
P4	C4	出生地	S5	李白出生于碎叶城。	李白	碎叶城
P5	C5	配偶	S6	习近平与彭丽媛是夫妻关系。	习近平	彭丽媛
```

## 参考

- [Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks](https://aclanthology.org/D15-1203/)
- [Neural Relation Extraction with Selective Attention over Instances](https://aclanthology.org/P16-1200/) 