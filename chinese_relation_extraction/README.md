# 中文关系抽取模型

基于OpenNRE框架开发的中文关系抽取模型，支持句子级和包级关系抽取。该项目使用中文BERT预训练模型和改进的实体编码方法，专门针对中文环境进行了优化。

## 项目特点

- 支持**句子级**和**包级**(远程监督)关系抽取
- 集成中文BERT预训练模型
- 提供多种实体表示方法：标准表示和注意力表示
- 支持**单标签**和**多标签**分类
- 包含完整的训练、评估和预测流程
- 提供交互式命令行预测界面

## 目录结构

```
chinese_relation_extraction/
├── configs/             # 配置文件目录
├── data/                # 数据集目录
├── models/              # 预训练模型和保存的模型目录
├── processors/          # 数据处理和编码器
│   ├── data_processor.py        # 数据处理器
│   └── chinese_bert_encoder.py  # 中文BERT编码器
├── models/              # 模型定义
│   └── chinese_re_model.py      # 中文关系抽取模型
├── utils.py             # 工具函数
├── train.py             # 训练脚本
├── predict.py           # 预测脚本
└── README.md            # 项目文档
```

## 安装

### 环境要求

- Python 3.6+
- PyTorch 1.6+
- Transformers 3.4+
- scikit-learn
- tqdm

### 安装依赖

```bash
pip install torch==1.6.0
pip install transformers==3.4.0
pip install scikit-learn==0.22.1
pip install tqdm
```

## 数据格式

### 句子级数据格式

训练数据和验证数据格式：
```
文本\t头实体\t尾实体\t关系类型
```

例如：
```
乔布斯于1976年在加利福尼亚创立了苹果公司。	乔布斯	苹果公司	创始人
```

### 包级数据格式(远程监督)

训练数据和验证数据格式：
```
头实体ID\t尾实体ID\t关系\t句子ID\t文本\t头实体\t尾实体
```

例如：
```
Q1	Q2	创始人	S1	乔布斯于1976年在加利福尼亚创立了苹果公司。	乔布斯	苹果公司
Q1	Q2	创始人	S2	苹果公司由史蒂夫·乔布斯、斯蒂夫·沃兹尼亚克和罗恩·韦恩创立。	乔布斯	苹果公司
```

### 关系标签文件格式

```
关系ID\t关系名称
```

例如：
```
0	创始人
1	总部位于
2	出生地
...
```

## 使用方法

### 训练模型

#### 句子级模型训练

```bash
python train.py \
    --mode sentence \
    --encoder_type standard \
    --train_file data/train.txt \
    --dev_file data/dev.txt \
    --relation_file data/relation.txt \
    --model_name_or_path bert-base-chinese \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --output_dir models/sentence_model
```

参数说明：
- `--mode`：训练模式，可选 "sentence" 或 "bag"
- `--encoder_type`：编码器类型，可选 "standard" 或 "attention"
- `--train_file`：训练数据文件路径
- `--dev_file`：验证数据文件路径
- `--relation_file`：关系标签文件路径
- `--model_name_or_path`：预训练模型名称或路径
- `--max_seq_length`：最大序列长度
- `--batch_size`：批次大小
- `--learning_rate`：学习率
- `--num_train_epochs`：训练轮数
- `--output_dir`：模型输出目录

#### 包级模型训练

```bash
python train.py \
    --mode bag \
    --bag_mode att \
    --train_file data/bag_train.txt \
    --dev_file data/bag_dev.txt \
    --relation_file data/relation.txt \
    --model_name_or_path bert-base-chinese \
    --max_seq_length 128 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --output_dir models/bag_model
```

额外参数：
- `--bag_mode`：包处理模式，可选 "one" 或 "att"

### 预测

#### 批量预测

```bash
python predict.py \
    --mode sentence \
    --encoder_type standard \
    --model_path models/sentence_model/checkpoint-epoch-3/model.pt \
    --model_name_or_path bert-base-chinese \
    --relation_file data/relation.txt \
    --input_file data/test.txt \
    --output_file data/test_result.txt
```

参数说明：
- `--mode`：预测模式，与训练时相同
- `--encoder_type`：编码器类型，与训练时相同
- `--model_path`：模型文件路径
- `--relation_file`：关系标签文件路径
- `--input_file`：输入文件路径，每行格式: "文本\t头实体\t尾实体"
- `--output_file`：输出文件路径，每行格式: "文本\t头实体\t尾实体\t关系\t置信度"

#### 交互式预测

```bash
python predict.py \
    --mode sentence \
    --encoder_type standard \
    --model_path models/sentence_model/checkpoint-epoch-3/model.pt \
    --model_name_or_path bert-base-chinese \
    --relation_file data/relation.txt \
    --interactive
```

在交互式界面中输入以下格式：
```
文本\t头实体\t尾实体
```

例如：
```
乔布斯于1976年在加利福尼亚创立了苹果公司。	乔布斯	苹果公司
```

## 模型架构

### 编码器

1. **ChineseBertEncoder**：
   - 基础中文BERT编码器，使用平均池化获取实体表示

2. **ChineseBertEntityAttentionEncoder**：
   - 增强版中文BERT编码器，使用注意力机制获取实体表示
   - 为每个实体计算注意力权重，获得更精确的实体表示

3. **ChineseBagBertEncoder**：
   - 包级中文BERT编码器，用于远程监督场景
   - 使用注意力机制聚合多个句子的表示

### 模型

1. **SentenceLevelModel**：
   - 句子级关系抽取模型
   - 结合句子表示和实体表示进行分类

2. **BagAttentionModel**：
   - 基于包注意力的关系抽取模型
   - 适用于远程监督场景，可以处理多个句子

3. **MultiLabelREModel**：
   - 多标签关系抽取模型
   - 适用于一个实体对可能具有多种关系的情况

## 样例

### 训练样例

#### 句子级模型训练
```bash
python train.py \
    --mode sentence \
    --encoder_type attention \
    --train_file data/train.txt \
    --dev_file data/dev.txt \
    --relation_file data/relation.txt \
    --model_name_or_path bert-base-chinese \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --output_dir models/sent_att_model
```

#### 包级模型训练
```bash
python train.py \
    --mode bag \
    --bag_mode att \
    --train_file data/bag_train.txt \
    --dev_file data/bag_dev.txt \
    --relation_file data/relation.txt \
    --model_name_or_path bert-base-chinese \
    --max_seq_length 128 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --output_dir models/bag_att_model
```

### 预测样例

```bash
python predict.py \
    --mode sentence \
    --encoder_type attention \
    --model_path models/sent_att_model/checkpoint-epoch-3/model.pt \
    --model_name_or_path bert-base-chinese \
    --relation_file data/relation.txt \
    --interactive
```

## 贡献和扩展

您可以通过以下方式扩展和改进这个项目：

1. 添加新的编码器，如基于其他中文预训练模型(RoBERTa, ERNIE等)的编码器
2. 添加新的模型架构，如图神经网络模型
3. 实现更多的数据增强和预处理方法
4. 添加更多的评估指标和分析工具

## 引用和致谢

本项目基于[OpenNRE](https://github.com/thunlp/OpenNRE)框架开发，针对中文关系抽取任务进行了特殊优化和扩展。感谢OpenNRE的开发者提供的优秀框架。

引用OpenNRE：
```
@inproceedings{han-etal-2019-opennre,
    title = "{O}pen{NRE}: An Open and Extensible Toolkit for Neural Relation Extraction",
    author = "Han, Xu and Gao, Tianyu and Yao, Yuan and Ye, Deming and Liu, Zhiyuan and Sun, Maosong",
    booktitle = "Proceedings of EMNLP-IJCNLP: System Demonstrations",
    year = "2019",
    url = "https://www.aclweb.org/anthology/D19-3029",
    doi = "10.18653/v1/D19-3029",
    pages = "169--174"
}
``` 