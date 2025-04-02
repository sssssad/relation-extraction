# coding=utf-8
"""
基于PCNN的远程监督关系抽取模型预测脚本
"""

import os
import sys
import argparse
import logging
import json
import torch
import numpy as np
from tqdm import tqdm

from processors.data_processor import load_relations, convert_example_to_feature
from processors.chinese_pcnn_encoder import ChinesePCNNEncoder, ChineseBagPCNNEncoder, ChineseWordTokenizer
from models.chinese_bag_pcnn_model import BagAttentionPCNNModel

# 设置日志
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_batch(args):
    """批量预测"""
    # 加载配置
    logger.info(f"从 {args.model_path} 加载模型")
    config_path = os.path.join(os.path.dirname(args.model_path), "config.json")
    
    # 如果有保存的配置，从配置中加载模型参数
    model_args = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model_args = json.load(f)
        logger.info(f"从 {config_path} 加载模型配置")
    
    # 更新/设置模型参数
    max_seq_length = model_args.get('max_seq_length', args.max_seq_length)
    hidden_size = model_args.get('hidden_size', args.hidden_size)
    word_size = model_args.get('word_size', args.word_size)
    position_size = model_args.get('position_size', args.position_size)
    kernel_size = model_args.get('kernel_size', args.kernel_size)
    padding_size = model_args.get('padding_size', args.padding_size)
    dropout = model_args.get('dropout', args.dropout)
    mask_entity = model_args.get('mask_entity', args.mask_entity)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载关系标签
    relation2id = load_relations(args.relation_file)
    id2relation = {v: k for k, v in relation2id.items()}
    num_relations = len(relation2id)
    logger.info(f"关系类别数量: {num_relations}")
    
    # 创建词汇表
    token2id = {}
    # 添加标准中文字符到词汇表
    for i in range(0x4e00, 0x9fff):  # 基本汉字范围
        char = chr(i)
        token2id[char] = len(token2id)
    
    # 添加特殊token
    for token in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
        token2id[token] = len(token2id)
    
    # 添加常见标点符号和数字
    for char in ',.?!;:\'\"()[]{}，。？！；：''""（）【】《》0123456789':
        token2id[char] = len(token2id)
    
    # 创建分词器
    tokenizer = ChineseWordTokenizer(vocab=token2id)
    
    # 创建编码器
    pcnn_encoder = ChinesePCNNEncoder(
        token2id=token2id,
        max_length=max_seq_length,
        hidden_size=hidden_size,
        word_size=word_size,
        position_size=position_size,
        blank_padding=True,
        kernel_size=kernel_size,
        padding_size=padding_size,
        dropout=dropout,
        mask_entity=mask_entity
    )
    
    # 创建包级编码器
    bag_encoder = ChineseBagPCNNEncoder(
        token2id=token2id,
        max_length=max_seq_length,
        hidden_size=hidden_size,
        word_size=word_size,
        position_size=position_size,
        blank_padding=True,
        kernel_size=kernel_size,
        padding_size=padding_size,
        dropout=dropout,
        mask_entity=mask_entity
    )
    
    # 创建模型
    model = BagAttentionPCNNModel(
        encoder=bag_encoder,
        num_relations=num_relations,
        dropout=dropout
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 加载测试数据
    test_examples = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 格式: 文本\t头实体\t尾实体
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            
            text, head, tail = parts
            test_examples.append({
                'text': text,
                'head': head,
                'tail': tail
            })
    
    logger.info(f"加载了 {len(test_examples)} 条测试数据")
    
    # 开始预测
    logger.info("开始预测...")
    results = []
    
    for example in tqdm(test_examples, desc="预测"):
        # 构建包
        bag = [{
            'text': example['text'],
            'h': {'pos': [example['text'].find(example['head']), 
                         example['text'].find(example['head']) + len(example['head'])]},
            't': {'pos': [example['text'].find(example['tail']), 
                         example['text'].find(example['tail']) + len(example['tail'])]}
        }]
        
        # 预测
        relation, confidence = model.infer(bag, relation2id, id2relation)
        
        results.append({
            'text': example['text'],
            'head': example['head'],
            'tail': example['tail'],
            'relation': relation,
            'confidence': float(confidence)
        })
    
    # 输出结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"{result['text']}\t{result['head']}\t{result['tail']}\t{result['relation']}\t{result['confidence']:.4f}\n")
    
    logger.info(f"预测结果已保存到 {args.output_file}")


def interactive_predict(args):
    """交互式预测"""
    # 加载配置
    logger.info(f"从 {args.model_path} 加载模型")
    config_path = os.path.join(os.path.dirname(args.model_path), "config.json")
    
    # 如果有保存的配置，从配置中加载模型参数
    model_args = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model_args = json.load(f)
        logger.info(f"从 {config_path} 加载模型配置")
    
    # 更新/设置模型参数
    max_seq_length = model_args.get('max_seq_length', args.max_seq_length)
    hidden_size = model_args.get('hidden_size', args.hidden_size)
    word_size = model_args.get('word_size', args.word_size)
    position_size = model_args.get('position_size', args.position_size)
    kernel_size = model_args.get('kernel_size', args.kernel_size)
    padding_size = model_args.get('padding_size', args.padding_size)
    dropout = model_args.get('dropout', args.dropout)
    mask_entity = model_args.get('mask_entity', args.mask_entity)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载关系标签
    relation2id = load_relations(args.relation_file)
    id2relation = {v: k for k, v in relation2id.items()}
    num_relations = len(relation2id)
    logger.info(f"关系类别数量: {num_relations}")
    
    # 创建词汇表
    token2id = {}
    # 添加标准中文字符到词汇表
    for i in range(0x4e00, 0x9fff):  # 基本汉字范围
        char = chr(i)
        token2id[char] = len(token2id)
    
    # 添加特殊token
    for token in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
        token2id[token] = len(token2id)
    
    # 添加常见标点符号和数字
    for char in ',.?!;:\'\"()[]{}，。？！；：''""（）【】《》0123456789':
        token2id[char] = len(token2id)
    
    # 创建分词器
    tokenizer = ChineseWordTokenizer(vocab=token2id)
    
    # 创建编码器
    pcnn_encoder = ChinesePCNNEncoder(
        token2id=token2id,
        max_length=max_seq_length,
        hidden_size=hidden_size,
        word_size=word_size,
        position_size=position_size,
        blank_padding=True,
        kernel_size=kernel_size,
        padding_size=padding_size,
        dropout=dropout,
        mask_entity=mask_entity
    )
    
    # 创建包级编码器
    bag_encoder = ChineseBagPCNNEncoder(
        token2id=token2id,
        max_length=max_seq_length,
        hidden_size=hidden_size,
        word_size=word_size,
        position_size=position_size,
        blank_padding=True,
        kernel_size=kernel_size,
        padding_size=padding_size,
        dropout=dropout,
        mask_entity=mask_entity
    )
    
    # 创建模型
    model = BagAttentionPCNNModel(
        encoder=bag_encoder,
        num_relations=num_relations,
        dropout=dropout
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 打印可用关系
    logger.info("可用关系类型:")
    for rel_id, rel_name in id2relation.items():
        logger.info(f"  {rel_id}: {rel_name}")
    
    # 交互式预测
    logger.info("\n开始交互式预测...")
    logger.info("输入格式: 文本\\t头实体\\t尾实体")
    logger.info("输入 'exit' 或 'quit' 退出")
    
    while True:
        user_input = input("\n请输入 (文本\\t头实体\\t尾实体): ")
        
        if user_input.lower() in ['exit', 'quit']:
            break
        
        try:
            parts = user_input.split('\t')
            if len(parts) != 3:
                logger.warning("输入格式错误，请使用 '文本\\t头实体\\t尾实体' 格式")
                continue
            
            text, head, tail = parts
            
            # 构建包
            bag = [{
                'text': text,
                'h': {'pos': [text.find(head), text.find(head) + len(head)]},
                't': {'pos': [text.find(tail), text.find(tail) + len(tail)]}
            }]
            
            # 预测
            relation, confidence = model.infer(bag, relation2id, id2relation)
            
            print(f"\n预测结果:")
            print(f"文本: {text}")
            print(f"头实体: {head}")
            print(f"尾实体: {tail}")
            print(f"关系: {relation}")
            print(f"置信度: {confidence:.4f}")
            
        except Exception as e:
            logger.error(f"处理输入时出错: {e}")


def main():
    parser = argparse.ArgumentParser()
    
    # 必要参数
    parser.add_argument("--model_path", required=True, type=str, help="模型文件路径")
    parser.add_argument("--relation_file", required=True, type=str, help="关系标签文件路径")
    
    # 模型参数
    parser.add_argument("--max_seq_length", default=128, type=int, help="最大序列长度")
    parser.add_argument("--hidden_size", default=230, type=int, help="隐藏层大小")
    parser.add_argument("--word_size", default=50, type=int, help="词嵌入大小")
    parser.add_argument("--position_size", default=5, type=int, help="位置嵌入大小")
    parser.add_argument("--kernel_size", default=3, type=int, help="卷积核大小")
    parser.add_argument("--padding_size", default=1, type=int, help="填充大小")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout比率")
    parser.add_argument("--mask_entity", action="store_true", help="是否掩盖实体")
    parser.add_argument("--no_cuda", action="store_true", help="不使用CUDA")
    
    # 批量预测参数
    parser.add_argument("--input_file", type=str, help="输入文件路径，每行格式: '文本\\t头实体\\t尾实体'")
    parser.add_argument("--output_file", type=str, help="输出文件路径，每行格式: '文本\\t头实体\\t尾实体\\t关系\\t置信度'")
    
    # 交互式预测
    parser.add_argument("--interactive", action="store_true", help="交互式预测模式")
    
    args = parser.parse_args()
    
    if args.interactive:
        # 交互式预测
        interactive_predict(args)
    else:
        # 批量预测
        if args.input_file is None or args.output_file is None:
            logger.error("批量预测模式需要指定 --input_file 和 --output_file")
            sys.exit(1)
        predict_batch(args)


if __name__ == "__main__":
    main() 