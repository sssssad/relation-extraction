# coding=utf-8
"""
中文关系抽取模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SentenceLevelModel(nn.Module):
    """句子级关系抽取模型"""
    
    def __init__(self, encoder, num_relations, dropout=0.1):
        """
        初始化模型
        
        Args:
            encoder: 句子编码器
            num_relations: 关系类别数量
            dropout: Dropout比率
        """
        super(SentenceLevelModel, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        
        # 输入特征维度 = 句子表示 + 实体表示
        input_size = encoder.hidden_size + 2 * encoder.hidden_size
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_relations)
        )
    
    def forward(self, input_ids, attention_mask=None, entity_mask=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            entity_mask: 实体位置掩码 [batch_size, seq_len]
        
        Returns:
            logits: 关系分类的logits [batch_size, num_relations]
        """
        # 获取句子表示和实体表示
        sent_repr, entity_repr = self.encoder(input_ids, attention_mask, entity_mask)
        
        # 拼接表示
        combined_repr = torch.cat([sent_repr, entity_repr], dim=1)
        combined_repr = self.dropout(combined_repr)
        
        # 分类
        logits = self.classifier(combined_repr)
        
        return logits
    
    def predict(self, input_ids, attention_mask=None, entity_mask=None):
        """预测关系"""
        logits = self.forward(input_ids, attention_mask, entity_mask)
        probs = F.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        return pred_labels, probs


class BagAttentionModel(nn.Module):
    """基于包注意力的关系抽取模型"""
    
    def __init__(self, encoder, num_relations, dropout=0.1):
        """
        初始化模型
        
        Args:
            encoder: 包级编码器
            num_relations: 关系类别数量
            dropout: Dropout比率
        """
        super(BagAttentionModel, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(encoder.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_relations)
        )
    
    def forward(self, input_ids, attention_mask=None, entity_mask=None, sentence_count=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, max_sent_num, seq_len]
            attention_mask: 注意力掩码 [batch_size, max_sent_num, seq_len]
            entity_mask: 实体位置掩码 [batch_size, max_sent_num, seq_len]
            sentence_count: 每个包中句子的数量 [batch_size]
        
        Returns:
            logits: 关系分类的logits [batch_size, num_relations]
        """
        # 获取包级表示
        bag_repr = self.encoder(input_ids, attention_mask, entity_mask, sentence_count)
        bag_repr = self.dropout(bag_repr)
        
        # 分类
        logits = self.classifier(bag_repr)
        
        return logits
    
    def predict(self, input_ids, attention_mask=None, entity_mask=None, sentence_count=None):
        """预测关系"""
        logits = self.forward(input_ids, attention_mask, entity_mask, sentence_count)
        probs = F.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        return pred_labels, probs


# 多标签关系抽取模型（用于一个句子可能有多个关系的情况）
class MultiLabelREModel(nn.Module):
    """多标签关系抽取模型"""
    
    def __init__(self, encoder, num_relations, dropout=0.1):
        """
        初始化模型
        
        Args:
            encoder: 句子编码器
            num_relations: 关系类别数量
            dropout: Dropout比率
        """
        super(MultiLabelREModel, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        
        # 输入特征维度 = 句子表示 + 实体表示
        input_size = encoder.hidden_size + 2 * encoder.hidden_size
        
        # 多标签分类器 (使用sigmoid而非softmax)
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_relations)
        )
        
        # 阈值
        self.threshold = 0.5
    
    def forward(self, input_ids, attention_mask=None, entity_mask=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            entity_mask: 实体位置掩码 [batch_size, seq_len]
        
        Returns:
            logits: 关系分类的logits [batch_size, num_relations]
        """
        # 获取句子表示和实体表示
        sent_repr, entity_repr = self.encoder(input_ids, attention_mask, entity_mask)
        
        # 拼接表示
        combined_repr = torch.cat([sent_repr, entity_repr], dim=1)
        combined_repr = self.dropout(combined_repr)
        
        # 分类
        logits = self.classifier(combined_repr)
        
        return logits
    
    def predict(self, input_ids, attention_mask=None, entity_mask=None):
        """预测多个关系"""
        logits = self.forward(input_ids, attention_mask, entity_mask)
        probs = torch.sigmoid(logits)
        # 大于阈值的关系被预测为正例
        pred_labels = (probs > self.threshold).float()
        return pred_labels, probs 