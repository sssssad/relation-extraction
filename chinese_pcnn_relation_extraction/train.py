# coding=utf-8
"""
基于PCNN的远程监督关系抽取模型训练脚本
"""

import os
import sys
import argparse
import logging
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report

from processors.data_processor import ChineseBagREDataset, get_data_loader, load_relations
from processors.chinese_pcnn_encoder import ChinesePCNNEncoder, ChineseBagPCNNEncoder, ChineseWordTokenizer
from models.chinese_bag_pcnn_model import BagAttentionPCNNModel

# 设置日志
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    """训练模型"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
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
    
    logger.info(f"词汇表大小: {len(token2id)}")
    
    # 创建分词器
    tokenizer = ChineseWordTokenizer(vocab=token2id)
    
    # 创建数据集
    train_dataset = ChineseBagREDataset(
        data_path=args.train_file,
        relation_path=args.relation_file,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        mode=args.bag_mode,
        vocab=token2id
    )
    
    dev_dataset = ChineseBagREDataset(
        data_path=args.dev_file,
        relation_path=args.relation_file,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        mode=args.bag_mode,
        vocab=token2id
    )
    
    # 为包级模型创建特殊的collate函数
    def bag_collate_fn(batch):
        if args.bag_mode == "one":
            # 单句模式
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
        
        # 注意力模式，需要特殊处理
        batch_size = len(batch)
        scope = []
        total_sents = 0
        
        # 计算范围
        for i, item in enumerate(batch):
            sent_num = item['sentence_count']
            scope.append((total_sents, total_sents + sent_num))
            total_sents += sent_num
        
        # 准备数据
        input_ids = []
        attention_mask = []
        entity_mask = []
        labels = []
        
        for item in batch:
            input_ids.extend(item['input_ids'])
            attention_mask.extend(item['attention_mask'])
            entity_mask.extend(item['entity_mask'])
            labels.append(item['label'])
        
        # 转换为张量
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        entity_mask = torch.tensor(entity_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        scope = torch.tensor(scope, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'entity_mask': entity_mask,
            'labels': labels,
            'scope': scope
        }
    
    # 创建数据加载器
    train_dataloader = get_data_loader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=bag_collate_fn
    )
    
    dev_dataloader = get_data_loader(
        dataset=dev_dataset, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        collate_fn=bag_collate_fn
    )
    
    # 创建编码器
    pcnn_encoder = ChinesePCNNEncoder(
        token2id=token2id,
        max_length=args.max_seq_length,
        hidden_size=args.hidden_size,
        word_size=args.word_size,
        position_size=args.position_size,
        blank_padding=True,
        kernel_size=args.kernel_size,
        padding_size=args.padding_size,
        dropout=args.dropout,
        mask_entity=args.mask_entity
    )
    
    # 创建包级编码器
    bag_encoder = ChineseBagPCNNEncoder(
        token2id=token2id,
        max_length=args.max_seq_length,
        hidden_size=args.hidden_size,
        word_size=args.word_size,
        position_size=args.position_size,
        blank_padding=True,
        kernel_size=args.kernel_size,
        padding_size=args.padding_size,
        dropout=args.dropout,
        mask_entity=args.mask_entity
    )
    
    # 创建模型
    model = BagAttentionPCNNModel(
        encoder=bag_encoder,
        num_relations=num_relations,
        dropout=args.dropout
    )
    
    # 将模型移至设备
    model.to(device)
    
    # 设置优化器和学习率调度器
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练
    logger.info("***** 开始训练 *****")
    logger.info(f"  训练轮数 = {args.num_train_epochs}")
    logger.info(f"  批次大小 = {args.batch_size}")
    logger.info(f"  学习率 = {args.learning_rate}")
    
    global_step = 0
    best_f1 = 0.0
    
    for epoch in range(args.num_train_epochs):
        logger.info(f"Epoch {epoch+1}/{args.num_train_epochs}")
        
        # 训练模式
        model.train()
        epoch_loss = 0.0
        
        # 训练循环
        progress_bar = tqdm(train_dataloader, desc="训练")
        for step, batch in enumerate(progress_bar):
            # 准备输入数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity_mask = batch['entity_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if 'scope' in batch:
                scope = batch['scope'].to(device)
            else:
                # 单句模式，创建虚拟scope
                batch_size = input_ids.size(0)
                scope = torch.tensor([[i, i+1] for i in range(batch_size)], dtype=torch.long).to(device)
            
            # 前向传播
            logits = model(
                token=input_ids,
                pos1=None,  # 由于数据处理方式不同，这里暂时不使用位置信息
                pos2=None, 
                mask=entity_mask,
                scope=scope
            )
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            
            # 更新进度条
            progress_bar.set_postfix(loss=loss.item())
            epoch_loss += loss.item()
            global_step += 1
            
            # 记录日志
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logger.info(f"  步骤 {global_step}: 损失 = {loss.item():.6f}")
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"  Epoch {epoch+1} 平均损失: {avg_epoch_loss:.6f}")
        
        # 评估
        results = evaluate(args, model, dev_dataloader, device, id2relation)
        
        # 保存最佳模型
        if results['micro_f1'] > best_f1:
            best_f1 = results['micro_f1']
            
            # 保存模型
            output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            logger.info(f"保存模型到 {output_dir}")
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
            
            # 保存配置
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                args_dict = vars(args)
                json.dump(args_dict, f, ensure_ascii=False, indent=4)
    
    logger.info("***** 训练完成 *****")
    logger.info(f"最佳 micro F1: {best_f1:.4f}")


def evaluate(args, model, dataloader, device, id2relation):
    """评估模型"""
    logger.info("***** 开始评估 *****")
    
    # 评估模式
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="评估"):
        # 准备输入数据
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        entity_mask = batch['entity_mask'].to(device)
        labels = batch['labels'].to(device)
        
        if 'scope' in batch:
            scope = batch['scope'].to(device)
        else:
            # 单句模式，创建虚拟scope
            batch_size = input_ids.size(0)
            scope = torch.tensor([[i, i+1] for i in range(batch_size)], dtype=torch.long).to(device)
        
        with torch.no_grad():
            # 前向传播
            pred_labels, _ = model.predict(
                token=input_ids,
                pos1=None,  # 由于数据处理方式不同，这里暂时不使用位置信息
                pos2=None, 
                mask=entity_mask,
                scope=scope
            )
        
        all_preds.extend(pred_labels.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro'
    )
    
    # 打印评估结果
    logger.info(f"精确率: {precision:.4f}")
    logger.info(f"召回率: {recall:.4f}")
    logger.info(f"F1: {f1:.4f}")
    
    # 打印分类报告
    target_names = [id2relation[i] for i in range(len(id2relation))]
    classification_rep = classification_report(
        all_labels, all_preds, target_names=target_names, digits=4
    )
    logger.info(f"分类报告:\n{classification_rep}")
    
    return {
        'micro_precision': precision,
        'micro_recall': recall,
        'micro_f1': f1
    }


def main():
    parser = argparse.ArgumentParser()
    
    # 必要参数
    parser.add_argument("--train_file", required=True, type=str, help="训练数据文件路径")
    parser.add_argument("--dev_file", required=True, type=str, help="验证数据文件路径")
    parser.add_argument("--relation_file", required=True, type=str, help="关系标签文件路径")
    parser.add_argument("--output_dir", required=True, type=str, help="模型输出目录")
    
    # 模型参数
    parser.add_argument("--bag_mode", default="att", type=str, choices=["one", "att"], help="包处理模式")
    parser.add_argument("--max_seq_length", default=128, type=int, help="最大序列长度")
    parser.add_argument("--hidden_size", default=230, type=int, help="隐藏层大小")
    parser.add_argument("--word_size", default=50, type=int, help="词嵌入大小")
    parser.add_argument("--position_size", default=5, type=int, help="位置嵌入大小")
    parser.add_argument("--kernel_size", default=3, type=int, help="卷积核大小")
    parser.add_argument("--padding_size", default=1, type=int, help="填充大小")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout比率")
    parser.add_argument("--mask_entity", action="store_true", help="是否掩盖实体")
    
    # 训练参数
    parser.add_argument("--batch_size", default=32, type=int, help="训练批次大小")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="评估批次大小")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="学习率")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="权重衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Adam优化器epsilon")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="梯度裁剪阈值")
    parser.add_argument("--num_train_epochs", default=30, type=int, help="训练轮数")
    parser.add_argument("--logging_steps", default=10, type=int, help="日志记录步数")
    parser.add_argument("--seed", default=42, type=int, help="随机种子")
    parser.add_argument("--no_cuda", action="store_true", help="不使用CUDA")
    
    args = parser.parse_args()
    
    # 训练模型
    train(args)


if __name__ == "__main__":
    main() 