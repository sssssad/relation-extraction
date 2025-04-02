# coding=utf-8
"""
中文关系抽取数据处理
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('..')
from utils import convert_example_to_feature, load_relations

class ChineseREDataset(Dataset):
    """中文关系抽取数据集"""
    
    def __init__(self, data_path, relation_path, tokenizer, max_length=128):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            relation_path: 关系文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.relation2id = load_relations(relation_path)
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.num_relations = len(self.relation2id)
        
        self.examples = []
        self.load_data(data_path)
    
    def load_data(self, data_path):
        """加载数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 格式: 文本\t头实体\t尾实体\t关系
                parts = line.split('\t')
                if len(parts) != 4:
                    continue
                
                text, head, tail, relation = parts
                
                if relation not in self.relation2id:
                    continue
                
                self.examples.append({
                    'text': text,
                    'head': head,
                    'tail': tail,
                    'relation': relation,
                    'relation_id': self.relation2id[relation]
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['text']
        head = example['head']
        tail = example['tail']
        label = example['relation_id']
        
        # 转换为特征
        feature = convert_example_to_feature(
            text=text, 
            head_entity=head, 
            tail_entity=tail, 
            tokenizer=self.tokenizer, 
            max_length=self.max_length
        )
        
        if feature is None:
            # 如果无法找到实体，返回一个默认特征
            # 在实际应用中可能需要更好的处理方式
            input_ids = self.tokenizer.encode(
                text, 
                max_length=self.max_length, 
                padding='max_length', 
                truncation=True
            )
            attention_mask = [1] * len(input_ids)
            entity_mask = [0] * len(input_ids)
            
            feature = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'entity_mask': entity_mask,
                'head_indices': [],
                'tail_indices': []
            }
        
        feature['label'] = label
        return feature

class BagREDataset(Dataset):
    """包级关系抽取数据集"""
    
    def __init__(self, data_path, relation_path, tokenizer, max_length=128, mode='one'):
        """
        初始化包级数据集
        
        Args:
            data_path: 数据文件路径
            relation_path: 关系文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            mode: 包处理模式，'one'表示每个包选一个句子，'att'表示使用注意力机制
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.relation2id = load_relations(relation_path)
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.num_relations = len(self.relation2id)
        self.mode = mode
        
        # 实体对到句子的映射
        self.entity_pair_to_sentences = {}
        
        self.load_data(data_path)
        self.index_bags()
    
    def load_data(self, data_path):
        """加载数据"""
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
    
    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        bag = self.bags[idx]
        label = bag['relation_id']
        sentences = bag['sentences']
        
        if self.mode == 'one':
            # 仅使用包中的一个句子（通常是第一个）
            sentence = sentences[0]
            feature = convert_example_to_feature(
                text=sentence['text'], 
                head_entity=sentence['head'], 
                tail_entity=sentence['tail'], 
                tokenizer=self.tokenizer, 
                max_length=self.max_length
            )
            
            if feature is None:
                # 默认处理
                text = sentence['text']
                input_ids = self.tokenizer.encode(
                    text, 
                    max_length=self.max_length, 
                    padding='max_length', 
                    truncation=True
                )
                attention_mask = [1] * len(input_ids)
                entity_mask = [0] * len(input_ids)
                
                feature = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'entity_mask': entity_mask,
                    'head_indices': [],
                    'tail_indices': []
                }
            
            feature['label'] = label
            return feature
        
        elif self.mode == 'att':
            # 处理包中的所有句子，供注意力机制使用
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
            
            if not features:
                # 如果包中没有有效特征，创建一个默认特征
                text = sentences[0]['text']
                input_ids = self.tokenizer.encode(
                    text, 
                    max_length=self.max_length, 
                    padding='max_length', 
                    truncation=True
                )
                attention_mask = [1] * len(input_ids)
                entity_mask = [0] * len(input_ids)
                
                features.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'entity_mask': entity_mask,
                    'head_indices': [],
                    'tail_indices': []
                })
            
            # 包装为包级特征
            bag_feature = {
                'input_ids': [f['input_ids'] for f in features],
                'attention_mask': [f['attention_mask'] for f in features],
                'entity_mask': [f['entity_mask'] for f in features],
                'sentence_count': len(features),
                'label': label
            }
            
            return bag_feature


def get_data_loader(dataset, batch_size, shuffle=True, collate_fn=None):
    """获取数据加载器"""
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn
    ) 