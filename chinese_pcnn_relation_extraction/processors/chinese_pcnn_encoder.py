# coding=utf-8
"""
中文PCNN编码器 - 分段卷积神经网络实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from transformers import BertTokenizer

class ChinesePCNNEncoder(nn.Module):
    """基于中文分段卷积神经网络的句子编码器"""
    
    def __init__(self, token2id, max_length=128, hidden_size=230, word_size=50,
                 position_size=5, blank_padding=True, word2vec=None,
                 kernel_size=3, padding_size=1, dropout=0.0, mask_entity=False):
        """
        初始化编码器
        
        Args:
            token2id: 词汇表字典，token->id映射
            max_length: 最大序列长度
            hidden_size: 隐藏层大小
            word_size: 词嵌入大小
            position_size: 位置嵌入大小
            blank_padding: 是否使用空白填充
            word2vec: 预训练词向量
            kernel_size: 卷积核大小
            padding_size: 填充大小
            dropout: Dropout比率
            mask_entity: 是否掩盖实体
        """
        super(ChinesePCNNEncoder, self).__init__()
        
        # 超参数
        self.token2id = token2id
        self.max_length = max_length
        self.num_token = len(token2id)
        self.num_position = max_length * 2
        self.mask_entity = mask_entity
        
        if word2vec is None:
            self.word_size = word_size
        else:
            self.word_size = word2vec.shape[-1]
            
        self.position_size = position_size
        # PCNN的隐藏层大小是原始隐藏层大小的3倍，因为有3个片段
        self.hidden_size = hidden_size * 3
        self.input_size = word_size + position_size * 2
        self.blank_padding = blank_padding
        
        # 添加特殊token
        if not '[UNK]' in self.token2id:
            self.token2id['[UNK]'] = len(self.token2id)
            self.num_token += 1
        if not '[PAD]' in self.token2id:
            self.token2id['[PAD]'] = len(self.token2id)
            self.num_token += 1
        
        # 词嵌入
        self.word_embedding = nn.Embedding(self.num_token, self.word_size)
        if word2vec is not None:
            logging.info("使用预训练词向量初始化词嵌入")
            word2vec = torch.from_numpy(word2vec)
            if self.num_token == len(word2vec) + 2:
                unk = torch.randn(1, self.word_size) / np.sqrt(self.word_size)
                blk = torch.zeros(1, self.word_size)
                self.word_embedding.weight.data.copy_(torch.cat([word2vec, unk, blk], 0))
            else:
                self.word_embedding.weight.data.copy_(word2vec)
        
        # 位置嵌入
        self.pos1_embedding = nn.Embedding(2 * max_length, self.position_size, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, self.position_size, padding_idx=0)
        
        # PCNN相关组件
        self.drop = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = F.relu
        
        # 卷积层
        self.conv = nn.Conv1d(self.input_size, hidden_size, self.kernel_size, padding=self.padding_size)
        
        # 分段池化所需的掩码嵌入
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.mask_embedding.weight.requires_grad = False
        self._minus = -100  # 用于掩盖不需要的部分
        
        # 分词器
        self.tokenizer = ChineseWordTokenizer(vocab=self.token2id, unk_token="[UNK]")
    
    def forward(self, token, pos1, pos2, mask):
        """
        前向传播
        
        Args:
            token: (B, L), token的索引
            pos1: (B, L), 相对于头实体的位置
            pos2: (B, L), 相对于尾实体的位置
            mask: (B, L), 分段掩码
        
        Returns:
            (B, H), 句子表示向量
        """
        # 检查张量大小
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():
            raise Exception("token、pos1和pos2的大小应该是(B, L)")
        
        # 将词嵌入和位置嵌入拼接
        x = torch.cat([self.word_embedding(token), 
                       self.pos1_embedding(pos1), 
                       self.pos2_embedding(pos2)], 2)  # (B, L, EMBED)
        x = x.transpose(1, 2)  # (B, EMBED, L)
        
        # 卷积
        x = self.conv(x)  # (B, H, L)
        
        # 分段池化
        mask = 1 - self.mask_embedding(mask).transpose(1, 2)  # (B, L) -> (B, L, 3) -> (B, 3, L)
        pool1 = F.max_pool1d(self.act(x + self._minus * mask[:, 0:1, :]), x.size(-1))  # (B, H, 1)
        pool2 = F.max_pool1d(self.act(x + self._minus * mask[:, 1:2, :]), x.size(-1))
        pool3 = F.max_pool1d(self.act(x + self._minus * mask[:, 2:3, :]), x.size(-1))
        
        # 拼接三个池化结果
        x = torch.cat([pool1, pool2, pool3], 1)  # (B, 3*H, 1)
        x = x.squeeze(2)  # (B, 3*H)
        x = self.drop(x)
        
        return x
    
    def tokenize(self, item):
        """
        分词处理
        
        Args:
            item: 包含文本和实体位置的字典
                {
                    'text': 文本,
                    'h': {'pos': [start, end], ...},
                    't': {'pos': [start, end], ...}
                }
        
        Returns:
            token_ids, pos1, pos2, mask
        """
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']
        
        # 句子转token
        if not is_token:
            # 确定头尾实体位置顺序
            if pos_head[0] > pos_tail[0]:
                pos_min, pos_max = [pos_tail, pos_head]
                rev = True
            else:
                pos_min, pos_max = [pos_head, pos_tail]
                rev = False
            
            # 分词
            sent_0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            sent_1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            sent_2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
            ent_0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            ent_1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            
            if self.mask_entity:
                ent_0 = ['[UNK]']
                ent_1 = ['[UNK]']
            
            tokens = sent_0 + ent_0 + sent_1 + ent_1 + sent_2
            
            # 更新头尾实体位置
            if rev:
                pos_tail = [len(sent_0), len(sent_0) + len(ent_0)]
                pos_head = [len(sent_0) + len(ent_0) + len(sent_1), len(sent_0) + len(ent_0) + len(sent_1) + len(ent_1)]
            else:
                pos_head = [len(sent_0), len(sent_0) + len(ent_0)]
                pos_tail = [len(sent_0) + len(ent_0) + len(sent_1), len(sent_0) + len(ent_0) + len(sent_1) + len(ent_1)]
        else:
            tokens = sentence
        
        # token转索引
        if self.blank_padding:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, self.max_length, self.token2id['[PAD]'], self.token2id['[UNK]'])
        else:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, unk_id=self.token2id['[UNK]'])
        
        # 计算位置索引
        pos1 = []
        pos2 = []
        pos1_in_index = min(pos_head[0], self.max_length)
        pos2_in_index = min(pos_tail[0], self.max_length)
        
        for i in range(len(tokens)):
            pos1.append(min(i - pos1_in_index + self.max_length, 2 * self.max_length - 1))
            pos2.append(min(i - pos2_in_index + self.max_length, 2 * self.max_length - 1))
        
        # 填充
        if self.blank_padding:
            while len(pos1) < self.max_length:
                pos1.append(0)
            while len(pos2) < self.max_length:
                pos2.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]
            pos1 = pos1[:self.max_length]
            pos2 = pos2[:self.max_length]
        
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)
        pos1 = torch.tensor(pos1).long().unsqueeze(0)  # (1, L)
        pos2 = torch.tensor(pos2).long().unsqueeze(0)  # (1, L)
        
        # 创建分段掩码
        mask = []
        pos_min = min(pos1_in_index, pos2_in_index)
        pos_max = max(pos1_in_index, pos2_in_index)
        
        for i in range(len(tokens)):
            if i <= pos_min:
                mask.append(1)  # 第一段：从句子开始到第一个实体
            elif i <= pos_max:
                mask.append(2)  # 第二段：从第一个实体到第二个实体
            else:
                mask.append(3)  # 第三段：从第二个实体到句子结束
        
        # 填充
        if self.blank_padding:
            while len(mask) < self.max_length:
                mask.append(0)
            mask = mask[:self.max_length]
        
        mask = torch.tensor(mask).long().unsqueeze(0)  # (1, L)
        
        return indexed_tokens, pos1, pos2, mask


class ChineseWordTokenizer:
    """中文单词分词器"""
    
    def __init__(self, vocab, unk_token="[UNK]"):
        self.vocab = vocab
        self.unk_token = unk_token
    
    def tokenize(self, text):
        """
        将文本分词为字符列表
        
        Args:
            text: 输入文本
        
        Returns:
            tokens列表
        """
        # 中文按字符分词
        tokens = [char for char in text]
        return tokens
    
    def convert_tokens_to_ids(self, tokens, max_length=None, pad_token_id=0, unk_token_id=1):
        """
        将tokens转换为ids
        
        Args:
            tokens: token列表
            max_length: 最大长度
            pad_token_id: 填充token的id
            unk_token_id: 未知token的id
        
        Returns:
            ids列表
        """
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(unk_token_id)
        
        # 填充或截断到指定长度
        if max_length is not None:
            if len(ids) < max_length:
                ids.extend([pad_token_id] * (max_length - len(ids)))
            else:
                ids = ids[:max_length]
        
        return ids


class ChineseBagPCNNEncoder(nn.Module):
    """包级PCNN编码器，用于远程监督场景"""
    
    def __init__(self, token2id, max_length=128, hidden_size=230, word_size=50,
                 position_size=5, blank_padding=True, word2vec=None,
                 kernel_size=3, padding_size=1, dropout=0.0, mask_entity=False):
        """
        初始化编码器
        
        Args:
            token2id: 词汇表字典，token->id映射
            max_length: 最大序列长度
            hidden_size: 隐藏层大小
            word_size: 词嵌入大小
            position_size: 位置嵌入大小
            blank_padding: 是否使用空白填充
            word2vec: 预训练词向量
            kernel_size: 卷积核大小
            padding_size: 填充大小
            dropout: Dropout比率
            mask_entity: 是否掩盖实体
        """
        super(ChineseBagPCNNEncoder, self).__init__()
        
        # 内部编码器
        self.pcnn_encoder = ChinesePCNNEncoder(
            token2id=token2id,
            max_length=max_length,
            hidden_size=hidden_size,
            word_size=word_size,
            position_size=position_size,
            blank_padding=blank_padding,
            word2vec=word2vec,
            kernel_size=kernel_size,
            padding_size=padding_size,
            dropout=dropout,
            mask_entity=mask_entity
        )
        
        # 包级注意力
        self.hidden_size = self.pcnn_encoder.hidden_size
        self.attention = nn.Parameter(torch.randn(self.hidden_size))
    
    def forward(self, token, pos1, pos2, mask, scope):
        """
        前向传播
        
        Args:
            token: (B, N, L), token的索引，B是批次大小，N是每个包中句子的数量
            pos1: (B, N, L), 相对于头实体的位置
            pos2: (B, N, L), 相对于尾实体的位置
            mask: (B, N, L), 分段掩码
            scope: (B, 2), 每个包的范围 [start, end)
        
        Returns:
            (B, H), 包级表示向量
        """
        # 获取批次大小和范围
        batch_size = len(scope)
        
        # 计算每个句子的表示
        if len(token.size()) == 3:
            # 已经有批次维度
            flat_token = token.reshape(-1, token.size(-1))  # (B*N, L)
            flat_pos1 = pos1.reshape(-1, pos1.size(-1))  # (B*N, L)
            flat_pos2 = pos2.reshape(-1, pos2.size(-1))  # (B*N, L)
            flat_mask = mask.reshape(-1, mask.size(-1))  # (B*N, L)
        else:
            # 没有批次维度，通常是推理时
            flat_token = token
            flat_pos1 = pos1
            flat_pos2 = pos2
            flat_mask = mask
        
        sent_repr = self.pcnn_encoder(flat_token, flat_pos1, flat_pos2, flat_mask)  # (B*N, H)
        
        # 准备包级表示
        bag_repr = torch.zeros(batch_size, self.hidden_size).to(sent_repr.device)  # (B, H)
        
        for i in range(batch_size):
            start, end = scope[i]
            if start < end:  # 确保包不为空
                # 获取该包的所有句子表示
                bag_sentences = sent_repr[start:end]  # (N, H)
                
                # 计算注意力分数
                attention_score = torch.matmul(bag_sentences, self.attention)  # (N)
                attention_score = F.softmax(attention_score, dim=0)  # (N)
                
                # 加权和
                bag_repr[i] = torch.matmul(attention_score.unsqueeze(0), bag_sentences).squeeze(0)  # (H)
        
        return bag_repr
    
    def tokenize(self, item):
        """使用内部编码器的tokenize方法"""
        return self.pcnn_encoder.tokenize(item) 