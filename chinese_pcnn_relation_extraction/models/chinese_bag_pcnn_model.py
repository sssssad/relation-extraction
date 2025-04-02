# coding=utf-8
"""
基于PCNN的远程监督关系抽取模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BagAttentionPCNNModel(nn.Module):
    """基于包注意力的PCNN关系抽取模型"""
    
    def __init__(self, encoder, num_relations, dropout=0.1):
        """
        初始化模型
        
        Args:
            encoder: 包级PCNN编码器
            num_relations: 关系类别数量
            dropout: Dropout比率
        """
        super(BagAttentionPCNNModel, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(encoder.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_relations)
        )
    
    def forward(self, token, pos1, pos2, mask, scope):
        """
        前向传播
        
        Args:
            token: (B*N, L) or (B, N, L), token的索引，B是批次大小，N是每个包中句子的数量
            pos1: (B*N, L) or (B, N, L), 相对于头实体的位置
            pos2: (B*N, L) or (B, N, L), 相对于尾实体的位置
            mask: (B*N, L) or (B, N, L), 分段掩码
            scope: (B, 2), 每个包的范围 [start, end)
        
        Returns:
            logits: 关系分类的logits [batch_size, num_relations]
        """
        # 获取包级表示
        bag_repr = self.encoder(token, pos1, pos2, mask, scope)
        bag_repr = self.dropout(bag_repr)
        
        # 分类
        logits = self.classifier(bag_repr)
        
        return logits
    
    def predict(self, token, pos1, pos2, mask, scope):
        """预测关系"""
        logits = self.forward(token, pos1, pos2, mask, scope)
        probs = F.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        return pred_labels, probs
    
    def infer(self, bag, rel2id, id2rel):
        """
        推理单个包中的关系
        
        Args:
            bag: 包含同一实体对的多个句子的列表
                [{
                  'text': 文本,
                  'h': {'pos': [start, end], ...},
                  't': {'pos': [start, end], ...}
                }]
            rel2id: 关系到id的映射字典
            id2rel: id到关系的映射字典
        
        Returns:
            (关系, 置信度)
        """
        self.eval()
        
        # 对包中的每个句子进行处理
        tokens, pos1s, pos2s, masks = [], [], [], []
        for item in bag:
            token, pos1, pos2, mask = self.encoder.tokenize(item)
            tokens.append(token)
            pos1s.append(pos1)
            pos2s.append(pos2)
            masks.append(mask)
        
        # 转换为张量
        tokens = torch.cat(tokens, 0)  # (N, L)
        pos1s = torch.cat(pos1s, 0)  # (N, L)
        pos2s = torch.cat(pos2s, 0)  # (N, L)
        masks = torch.cat(masks, 0)  # (N, L)
        scope = torch.tensor([[0, len(bag)]]).long()  # (1, 2)
        
        # 前向传播
        logits = self.forward(tokens, pos1s, pos2s, masks, scope)  # (1, num_relations)
        probs = F.softmax(logits, dim=1).squeeze(0)  # (num_relations)
        
        # 获取最可能的关系及其置信度
        score, pred = probs.max(0)
        score = score.item()
        pred = pred.item()
        
        if id2rel is not None:
            pred = id2rel[pred]
        
        return pred, score 