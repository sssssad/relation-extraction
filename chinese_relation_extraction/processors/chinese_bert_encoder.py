# coding=utf-8
"""
中文BERT编码器
"""

import torch
import torch.nn as nn
from transformers import BertModel

class ChineseBertEncoder(nn.Module):
    """基于中文BERT的句子编码器"""
    
    def __init__(self, pretrain_path, max_length=128):
        """
        初始化编码器
        
        Args:
            pretrain_path: 预训练模型路径或名称，如 'bert-base-chinese'
            max_length: 最大序列长度
        """
        super(ChineseBertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.hidden_size = self.bert.config.hidden_size
    
    def forward(self, input_ids, attention_mask=None, entity_mask=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            entity_mask: 实体位置掩码 [batch_size, seq_len]，1表示头实体，2表示尾实体
        
        Returns:
            实体表示和全句表示
        """
        # 使用BERT获取隐藏状态
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        # 提取头尾实体表示方法1：使用实体掩码提取
        if entity_mask is not None:
            batch_size = input_ids.size(0)
            
            # 提取头实体表示
            head_mask = (entity_mask == 1).float()  # [batch_size, seq_len]
            head_mask = head_mask.unsqueeze(2).expand_as(last_hidden_state)  # [batch_size, seq_len, hidden_size]
            head_sum = torch.sum(last_hidden_state * head_mask, dim=1)  # [batch_size, hidden_size]
            head_count = torch.sum(entity_mask == 1, dim=1).float().unsqueeze(1)  # [batch_size, 1]
            head_count = torch.clamp(head_count, min=1)
            head_repr = head_sum / head_count  # [batch_size, hidden_size]
            
            # 提取尾实体表示
            tail_mask = (entity_mask == 2).float()  # [batch_size, seq_len]
            tail_mask = tail_mask.unsqueeze(2).expand_as(last_hidden_state)  # [batch_size, seq_len, hidden_size]
            tail_sum = torch.sum(last_hidden_state * tail_mask, dim=1)  # [batch_size, hidden_size]
            tail_count = torch.sum(entity_mask == 2, dim=1).float().unsqueeze(1)  # [batch_size, 1]
            tail_count = torch.clamp(tail_count, min=1)
            tail_repr = tail_sum / tail_count  # [batch_size, hidden_size]
            
            # 连接头尾实体表示和[CLS]表示
            entity_repr = torch.cat([head_repr, tail_repr], dim=1)  # [batch_size, 2*hidden_size]
            
            return pooled_output, entity_repr
        
        # 提取头尾实体表示方法2：直接使用[CLS]表示
        return pooled_output, pooled_output


class ChineseBertEntityAttentionEncoder(nn.Module):
    """基于中文BERT的实体注意力编码器"""
    
    def __init__(self, pretrain_path, max_length=128):
        """
        初始化编码器
        
        Args:
            pretrain_path: 预训练模型路径或名称，如 'bert-base-chinese'
            max_length: 最大序列长度
        """
        super(ChineseBertEntityAttentionEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.hidden_size = self.bert.config.hidden_size
        
        # 头尾实体注意力层
        self.head_attention = nn.Linear(self.hidden_size, 1)
        self.tail_attention = nn.Linear(self.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask=None, entity_mask=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            entity_mask: 实体位置掩码 [batch_size, seq_len]，1表示头实体，2表示尾实体
        
        Returns:
            实体表示和全句表示
        """
        # 使用BERT获取隐藏状态
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        batch_size = input_ids.size(0)
        
        # 使用注意力机制提取实体表示
        if entity_mask is not None:
            # 创建头实体和尾实体的掩码
            head_mask = (entity_mask == 1).float()  # [batch_size, seq_len]
            head_mask = head_mask * attention_mask.float()  # 应用注意力掩码
            
            tail_mask = (entity_mask == 2).float()  # [batch_size, seq_len]
            tail_mask = tail_mask * attention_mask.float()  # 应用注意力掩码
            
            # 计算头实体注意力分数
            head_attn = self.head_attention(last_hidden_state).squeeze(-1)  # [batch_size, seq_len]
            head_attn = head_attn.masked_fill((1 - head_mask).bool(), -1e9)
            head_attn = torch.softmax(head_attn, dim=1)
            head_attn = head_attn.unsqueeze(2)  # [batch_size, seq_len, 1]
            
            # 计算尾实体注意力分数
            tail_attn = self.tail_attention(last_hidden_state).squeeze(-1)  # [batch_size, seq_len]
            tail_attn = tail_attn.masked_fill((1 - tail_mask).bool(), -1e9)
            tail_attn = torch.softmax(tail_attn, dim=1)
            tail_attn = tail_attn.unsqueeze(2)  # [batch_size, seq_len, 1]
            
            # 计算加权和得到实体表示
            head_repr = torch.sum(head_attn * last_hidden_state, dim=1)  # [batch_size, hidden_size]
            tail_repr = torch.sum(tail_attn * last_hidden_state, dim=1)  # [batch_size, hidden_size]
            
            # 对于没有实体token的情况，使用[CLS]表示
            head_exists = (torch.sum(head_mask, dim=1) > 0).float().unsqueeze(1)  # [batch_size, 1]
            tail_exists = (torch.sum(tail_mask, dim=1) > 0).float().unsqueeze(1)  # [batch_size, 1]
            
            head_repr = head_exists * head_repr + (1 - head_exists) * pooled_output
            tail_repr = tail_exists * tail_repr + (1 - tail_exists) * pooled_output
            
            # 连接头尾实体表示
            entity_repr = torch.cat([head_repr, tail_repr], dim=1)  # [batch_size, 2*hidden_size]
            
            return pooled_output, entity_repr
        
        # 如果没有实体掩码，直接使用[CLS]表示
        return pooled_output, pooled_output


# 包级编码器(用于远程监督)
class ChineseBagBertEncoder(nn.Module):
    """包级BERT编码器，用于处理包含多个句子的包"""
    
    def __init__(self, pretrain_path, max_length=128):
        """
        初始化编码器
        
        Args:
            pretrain_path: 预训练模型路径或名称，如 'bert-base-chinese'
            max_length: 最大序列长度
        """
        super(ChineseBagBertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.hidden_size = self.bert.config.hidden_size
        
        # 用于计算句子级注意力的查询向量
        self.query = nn.Parameter(torch.randn(self.hidden_size))
        # 注意力转换
        self.attention = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self, input_ids, attention_mask=None, entity_mask=None, sentence_count=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, max_sent_num, seq_len]
            attention_mask: 注意力掩码 [batch_size, max_sent_num, seq_len]
            entity_mask: 实体位置掩码 [batch_size, max_sent_num, seq_len]
            sentence_count: 每个包中句子的数量 [batch_size]
        
        Returns:
            包级表示
        """
        if sentence_count is None:
            # 单句模式，直接使用普通编码器
            return self.encode_single(input_ids, attention_mask, entity_mask)
        
        # 包级处理
        batch_size = input_ids.size(0)
        max_sent_num = input_ids.size(1)
        
        # 将所有句子平展处理
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))  # [batch_size*max_sent_num, seq_len]
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))  # [batch_size*max_sent_num, seq_len]
        
        # 使用BERT处理所有句子
        outputs = self.bert(flat_input_ids, attention_mask=flat_attention_mask)
        flat_pooled_output = outputs.pooler_output  # [batch_size*max_sent_num, hidden_size]
        
        # 重塑为包的形状
        sentence_repr = flat_pooled_output.view(batch_size, max_sent_num, -1)  # [batch_size, max_sent_num, hidden_size]
        
        # 创建包级掩码，标记有效句子
        bag_mask = torch.zeros(batch_size, max_sent_num).to(input_ids.device)
        for i in range(batch_size):
            bag_mask[i, :sentence_count[i]] = 1
        
        # 计算句子级注意力
        attention_input = self.attention(sentence_repr)  # [batch_size, max_sent_num, hidden_size]
        attention_score = torch.matmul(attention_input, self.query)  # [batch_size, max_sent_num]
        
        # 应用掩码
        attention_score = attention_score.masked_fill((1 - bag_mask).bool(), -1e9)
        attention_weight = torch.softmax(attention_score, dim=1).unsqueeze(2)  # [batch_size, max_sent_num, 1]
        
        # 计算加权和
        bag_repr = torch.sum(attention_weight * sentence_repr, dim=1)  # [batch_size, hidden_size]
        
        return bag_repr
    
    def encode_single(self, input_ids, attention_mask=None, entity_mask=None):
        """单句编码"""
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        return pooled_output 