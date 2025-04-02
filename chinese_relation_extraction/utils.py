# coding=utf-8
"""
中文关系抽取工具函数
"""
import os
import json
import torch
import numpy as np
from transformers import BertTokenizer

def load_json(filename):
    """加载json文件"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filename):
    """保存json文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_relations(filename):
    """加载关系标签文件"""
    relations = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_id, rel_name = line.split('\t')
            relations[rel_name] = int(rel_id)
    return relations

def find_entity_position(text, entity):
    """
    在文本中查找实体位置
    返回: (start_pos, end_pos)
    """
    start_pos = text.find(entity)
    if start_pos == -1:
        return None
    end_pos = start_pos + len(entity)
    return (start_pos, end_pos)

def convert_example_to_feature(text, head_entity, tail_entity, tokenizer, max_length=128):
    """
    将原始文本和实体转换为模型输入特征
    """
    # 查找实体位置
    head_pos = find_entity_position(text, head_entity)
    tail_pos = find_entity_position(text, tail_entity)
    
    if head_pos is None or tail_pos is None:
        return None
    
    # 对文本进行分词，不使用offset_mapping
    encoded = tokenizer(text, max_length=max_length, 
                       truncation=True, padding='max_length')
    
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    
    # 使用另一种方法定位实体
    # 我们分别对文本和实体进行分词，然后匹配子序列
    head_tokens = tokenizer.tokenize(head_entity)
    tail_tokens = tokenizer.tokenize(tail_entity)
    text_tokens = tokenizer.tokenize(text)
    
    # 检查分词后的文本是否超过最大长度
    if len(text_tokens) > max_length - 2:  # 考虑[CLS]和[SEP]
        text_tokens = text_tokens[:max_length - 2]
    
    # 查找实体对应的token位置
    head_token_indices = []
    tail_token_indices = []
    
    # 查找头实体
    for i in range(len(text_tokens) - len(head_tokens) + 1):
        match = True
        for j in range(len(head_tokens)):
            if i + j >= len(text_tokens) or text_tokens[i + j] != head_tokens[j]:
                match = False
                break
        if match:
            # 找到匹配，需要考虑[CLS]的偏移
            for j in range(len(head_tokens)):
                head_token_indices.append(i + j + 1)  # +1是因为[CLS]
    
    # 查找尾实体
    for i in range(len(text_tokens) - len(tail_tokens) + 1):
        match = True
        for j in range(len(tail_tokens)):
            if i + j >= len(text_tokens) or text_tokens[i + j] != tail_tokens[j]:
                match = False
                break
        if match:
            # 找到匹配，需要考虑[CLS]的偏移
            for j in range(len(tail_tokens)):
                tail_token_indices.append(i + j + 1)  # +1是因为[CLS]
    
    # 创建实体掩码 (1表示头实体，2表示尾实体，0表示其他)
    entity_mask = [0] * len(input_ids)
    
    # 确保索引在范围内
    head_token_indices = [idx for idx in head_token_indices if idx < len(input_ids)]
    tail_token_indices = [idx for idx in tail_token_indices if idx < len(input_ids)]
    
    for idx in head_token_indices:
        entity_mask[idx] = 1
    for idx in tail_token_indices:
        entity_mask[idx] = 2
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'entity_mask': entity_mask,
        'head_indices': head_token_indices,
        'tail_indices': tail_token_indices
    }

def collate_fn(batch):
    """批处理函数，用于DataLoader"""
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