# coding=utf-8
"""
中文关系抽取数据处理模块
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import json
import logging

logger = logging.getLogger(__name__)

def load_relations(relation_file):
    """加载关系标签文件"""
    relation2id = {}
    
    with open(relation_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            
            rel_id, rel_name = parts
            relation2id[rel_name] = int(rel_id)
    
    return relation2id

def convert_example_to_feature(text, head_entity, tail_entity, tokenizer, max_length):
    """
    将文本和实体转换为模型特征
    
    Args:
        text: 文本
        head_entity: 头实体文本
        tail_entity: 尾实体文本
        tokenizer: 分词器
        max_length: 最大序列长度
    
    Returns:
        特征字典
    """
    # 找到实体在文本中的位置
    head_start = text.find(head_entity)
    tail_start = text.find(tail_entity)
    
    if head_start == -1 or tail_start == -1:
        return None
    
    head_end = head_start + len(head_entity)
    tail_end = tail_start + len(tail_entity)
    
    # 分词并获取token ids
    tokens = [char for char in text]  # 中文按字符分词
    input_ids = []
    
    for token in tokens:
        if token in tokenizer.vocab:
            input_ids.append(tokenizer.vocab[token])
        else:
            input_ids.append(tokenizer.vocab.get('[UNK]', 1))
    
    # 创建实体掩码
    entity_mask = [0] * len(tokens)
    for i in range(head_start, head_end):
        if i < len(entity_mask):
            entity_mask[i] = 1  # 头实体
    
    for i in range(tail_start, tail_end):
        if i < len(entity_mask):
            entity_mask[i] = 2  # 尾实体
    
    # 处理序列长度
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        entity_mask = entity_mask[:max_length]
    
    # 创建注意力掩码
    attention_mask = [1] * len(input_ids)
    
    # 填充到最大长度
    padding_length = max_length - len(input_ids)
    input_ids.extend([0] * padding_length)
    attention_mask.extend([0] * padding_length)
    entity_mask.extend([0] * padding_length)
    
    # 记录头尾实体位置
    head_indices = list(range(head_start, min(head_end, max_length)))
    tail_indices = list(range(tail_start, min(tail_end, max_length)))
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'entity_mask': entity_mask,
        'head_indices': head_indices,
        'tail_indices': tail_indices
    }

def collate_fn(batch):
    """
    数据批次整理函数
    
    Args:
        batch: 样本批次
    
    Returns:
        整理后的批次
    """
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
    entity_mask = torch.tensor([item['entity_mask'] for item in batch], dtype=torch.long)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'entity_mask': entity_mask,
        'labels': labels
    }

class ChineseBagREDataset(Dataset):
    """中文包级关系抽取数据集"""
    
    def __init__(self, data_path, relation_path, tokenizer, max_length=128, mode='att', vocab=None):
        """
        初始化包级数据集
        
        Args:
            data_path: 数据文件路径
            relation_path: 关系文件路径
            tokenizer: 分词器对象或词汇表字典
            max_length: 最大序列长度
            mode: 包处理模式，'one'表示每个包选一个句子，'att'表示使用注意力机制
            vocab: 词汇表字典，如果tokenizer是分词器对象则不需要
        """
        self.max_length = max_length
        self.relation2id = load_relations(relation_path)
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.num_relations = len(self.relation2id)
        self.mode = mode
        
        # 处理tokenizer
        self.tokenizer = tokenizer
        self.vocab = vocab if vocab else tokenizer.vocab if hasattr(tokenizer, 'vocab') else {}
        
        # 实体对到句子的映射
        self.entity_pair_to_sentences = {}
        
        # 加载数据
        self.load_data(data_path)
        self.index_bags()
    
    def load_data(self, data_path):
        """加载数据"""
        logger.info(f"从 {data_path} 加载数据...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 格式: 头实体ID\t尾实体ID\t关系\t句子ID\t文本\t头实体\t尾实体
                parts = line.split('\t')
                if len(parts) != 7:
                    continue
                
                head_id, tail_id, relation, sent_id, text, head, tail = parts
                
                if relation not in self.relation2id:
                    continue
                
                # 构建实体对key
                key = (head_id, tail_id, relation)
                
                if key not in self.entity_pair_to_sentences:
                    self.entity_pair_to_sentences[key] = []
                
                self.entity_pair_to_sentences[key].append({
                    'text': text,
                    'head': head,
                    'tail': tail,
                    'sent_id': sent_id,
                    'relation': relation,
                    'relation_id': self.relation2id[relation]
                })
        
        logger.info(f"加载了 {len(self.entity_pair_to_sentences)} 个实体对")
    
    def index_bags(self):
        """将数据整理成包的形式"""
        self.bags = []
        for key, sentences in self.entity_pair_to_sentences.items():
            head_id, tail_id, relation = key
            self.bags.append({
                'head_id': head_id,
                'tail_id': tail_id,
                'relation': relation,
                'relation_id': self.relation2id[relation],
                'sentences': sentences
            })
        
        logger.info(f"共有 {len(self.bags)} 个包")
    
    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        bag = self.bags[idx]
        label = bag['relation_id']
        sentences = bag['sentences']
        
        if self.mode == 'one':
            # 仅使用包中的一个句子（通常是第一个）
            sentence = sentences[0]
            
            # 转换为特征
            feature = convert_example_to_feature(
                text=sentence['text'], 
                head_entity=sentence['head'], 
                tail_entity=sentence['tail'], 
                tokenizer=self.tokenizer, 
                max_length=self.max_length
            )
            
            if feature is None:
                # 默认处理
                feature = {
                    'input_ids': [0] * self.max_length,
                    'attention_mask': [0] * self.max_length,
                    'entity_mask': [0] * self.max_length,
                    'head_indices': [],
                    'tail_indices': []
                }
            
            return {
                'input_ids': feature['input_ids'],
                'attention_mask': feature['attention_mask'],
                'entity_mask': feature['entity_mask'],
                'label': label
            }
        
        elif self.mode == 'att':
            # 处理包中的所有句子用于注意力机制
            features = []
            
            for sentence in sentences:
                feature = convert_example_to_feature(
                    text=sentence['text'], 
                    head_entity=sentence['head'], 
                    tail_entity=sentence['tail'], 
                    tokenizer=self.tokenizer, 
                    max_length=self.max_length
                )
                
                if feature is not None:
                    features.append(feature)
            
            # 如果没有有效的特征，添加一个空特征
            if not features:
                features.append({
                    'input_ids': [0] * self.max_length,
                    'attention_mask': [0] * self.max_length,
                    'entity_mask': [0] * self.max_length,
                    'head_indices': [],
                    'tail_indices': []
                })
            
            # 返回所有句子特征和句子数量
            return {
                'input_ids': [feature['input_ids'] for feature in features],
                'attention_mask': [feature['attention_mask'] for feature in features],
                'entity_mask': [feature['entity_mask'] for feature in features],
                'sentence_count': len(features),
                'label': label
            }
        
        else:
            raise ValueError(f"不支持的模式: {self.mode}")

def get_data_loader(dataset, batch_size, shuffle=True, collate_fn=None):
    """创建数据加载器"""
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    ) 