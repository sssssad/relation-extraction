# coding=utf-8
"""
中文关系抽取模型训练脚本
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
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report

from processors.data_processor import ChineseREDataset, BagREDataset, get_data_loader
from processors.chinese_bert_encoder import ChineseBertEncoder, ChineseBertEntityAttentionEncoder, ChineseBagBertEncoder
from models.chinese_re_model import SentenceLevelModel, BagAttentionModel, MultiLabelREModel
from utils import load_relations, collate_fn

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
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    
    # 加载关系标签
    relation2id = load_relations(args.relation_file)
    num_relations = len(relation2id)
    logger.info(f"关系类别数量: {num_relations}")
    
    # 加载数据集
    if args.mode == "sentence":
        train_dataset = ChineseREDataset(
            data_path=args.train_file,
            relation_path=args.relation_file,
            tokenizer=tokenizer,
            max_length=args.max_seq_length
        )
        
        dev_dataset = ChineseREDataset(
            data_path=args.dev_file,
            relation_path=args.relation_file,
            tokenizer=tokenizer,
            max_length=args.max_seq_length
        )
        
        # 创建数据加载器
        train_dataloader = get_data_loader(
            dataset=train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        dev_dataloader = get_data_loader(
            dataset=dev_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        # 创建编码器
        if args.encoder_type == "standard":
            encoder = ChineseBertEncoder(
                pretrain_path=args.model_name_or_path,
                max_length=args.max_seq_length
            )
        elif args.encoder_type == "attention":
            encoder = ChineseBertEntityAttentionEncoder(
                pretrain_path=args.model_name_or_path,
                max_length=args.max_seq_length
            )
        else:
            raise ValueError(f"不支持的编码器类型: {args.encoder_type}")
        
        # 创建模型
        if args.multi_label:
            model = MultiLabelREModel(
                encoder=encoder,
                num_relations=num_relations,
                dropout=args.dropout
            )
        else:
            model = SentenceLevelModel(
                encoder=encoder,
                num_relations=num_relations,
                dropout=args.dropout
            )
    
    elif args.mode == "bag":
        train_dataset = BagREDataset(
            data_path=args.train_file,
            relation_path=args.relation_file,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            mode=args.bag_mode
        )
        
        dev_dataset = BagREDataset(
            data_path=args.dev_file,
            relation_path=args.relation_file,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            mode=args.bag_mode
        )
        
        # 为包级模型创建特殊的collate函数
        def bag_collate_fn(batch):
            if args.bag_mode == "one":
                # 单句模式，直接使用普通collate_fn
                return collate_fn(batch)
            
            # 注意力模式，需要特殊处理
            max_sent_num = max([item['sentence_count'] for item in batch])
            batch_size = len(batch)
            
            # 创建填充张量
            input_ids = torch.zeros(batch_size, max_sent_num, args.max_seq_length, dtype=torch.long)
            attention_mask = torch.zeros(batch_size, max_sent_num, args.max_seq_length, dtype=torch.long)
            entity_mask = torch.zeros(batch_size, max_sent_num, args.max_seq_length, dtype=torch.long)
            sentence_count = torch.tensor([item['sentence_count'] for item in batch], dtype=torch.long)
            labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
            
            # 填充数据
            for i, item in enumerate(batch):
                sent_count = item['sentence_count']
                input_ids[i, :sent_count] = torch.tensor(item['input_ids'][:sent_count], dtype=torch.long)
                attention_mask[i, :sent_count] = torch.tensor(item['attention_mask'][:sent_count], dtype=torch.long)
                entity_mask[i, :sent_count] = torch.tensor(item['entity_mask'][:sent_count], dtype=torch.long)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'entity_mask': entity_mask,
                'sentence_count': sentence_count,
                'labels': labels
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
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=bag_collate_fn
        )
        
        # 创建包级编码器和模型
        encoder = ChineseBagBertEncoder(
            pretrain_path=args.model_name_or_path,
            max_length=args.max_seq_length
        )
        
        model = BagAttentionModel(
            encoder=encoder,
            num_relations=num_relations,
            dropout=args.dropout
        )
    
    else:
        raise ValueError(f"不支持的模式: {args.mode}")
    
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
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    # 计算总训练步数
    t_total = len(train_dataloader) * args.num_train_epochs
    
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )
    
    # 记录训练信息
    logger.info("***** 开始训练 *****")
    logger.info(f"  模式 = {args.mode}")
    logger.info(f"  批次大小 = {args.batch_size}")
    logger.info(f"  训练轮数 = {args.num_train_epochs}")
    logger.info(f"  学习率 = {args.learning_rate}")
    logger.info(f"  总训练步数 = {t_total}")
    
    # 训练循环
    global_step = 0
    best_f1 = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"第{epoch+1}轮训练")):
            # 将数据移至设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity_mask = batch['entity_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            if args.mode == "bag" and args.bag_mode == "att":
                sentence_count = batch['sentence_count'].to(device)
                logits = model(input_ids, attention_mask, entity_mask, sentence_count)
            else:
                logits = model(input_ids, attention_mask, entity_mask)
            
            # 计算损失
            if args.multi_label:
                # 多标签分类使用BCE损失
                loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
            else:
                # 单标签分类使用交叉熵损失
                loss = torch.nn.CrossEntropyLoss()(logits, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            global_step += 1
            epoch_loss += loss.item()
            
            # 每隔一定步数在验证集上评估
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logger.info(f"  步数 {global_step} / {t_total}, 损失: {loss.item():.6f}")
            
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                # 在验证集上评估
                results = evaluate(args, model, dev_dataloader, device)
                
                # 如果是最佳模型，保存
                if results['f1'] > best_f1:
                    best_f1 = results['f1']
                    logger.info(f"  找到新的最佳模型！F1: {best_f1:.4f}")
                    
                    # 保存模型
                    output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    # 保存模型状态
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.pt"))
                    
                    # 保存训练参数
                    with open(os.path.join(output_dir, "training_args.json"), 'w') as f:
                        json.dump(vars(args), f, ensure_ascii=False, indent=2)
                
                model.train()
        
        # 每轮结束后在验证集上评估
        logger.info(f"  第{epoch+1}轮平均损失: {epoch_loss / len(train_dataloader):.6f}")
        results = evaluate(args, model, dev_dataloader, device)
        
        # 如果是最佳模型，保存
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            logger.info(f"  找到新的最佳模型！F1: {best_f1:.4f}")
            
            # 保存模型
            output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 保存模型状态
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.pt"))
            
            # 保存训练参数
            with open(os.path.join(output_dir, "training_args.json"), 'w') as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=2)
            
            # 同时保存到best_model目录
            best_model_dir = os.path.join(args.output_dir, "best_model")
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)
            
            torch.save(model_to_save.state_dict(), os.path.join(best_model_dir, "model.pt"))
            with open(os.path.join(best_model_dir, "training_args.json"), 'w') as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=2)
    
    # 训练结束
    logger.info("***** 训练完成 *****")
    logger.info(f"  最佳F1分数: {best_f1:.4f}")


def evaluate(args, model, eval_dataloader, device):
    """在验证集上评估模型"""
    
    # 设置为评估模式
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in tqdm(eval_dataloader, desc="评估中"):
        # 将数据移至设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        entity_mask = batch['entity_mask'].to(device)
        labels = batch['labels']
        
        with torch.no_grad():
            # 前向传播
            if args.mode == "bag" and args.bag_mode == "att":
                sentence_count = batch['sentence_count'].to(device)
                logits = model(input_ids, attention_mask, entity_mask, sentence_count)
            else:
                logits = model(input_ids, attention_mask, entity_mask)
            
            # 获取预测结果
            if args.multi_label:
                # 多标签情况，使用sigmoid阈值
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float().cpu().numpy()
            else:
                # 单标签情况，取最大值
                preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    
    # 计算评估指标
    if args.multi_label:
        # 多标签评估
        precision, recall, f1, _ = precision_recall_fscore_support(
            np.array(all_labels), 
            np.array(all_preds),
            average='micro'
        )
        
        logger.info(f"  多标签评估 - 准确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
    else:
        # 单标签评估
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, 
            all_preds,
            average='macro'
        )
        
        logger.info(f"  单标签评估 - 准确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
        logger.info("\n" + classification_report(all_labels, all_preds))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser()
    
    # 基础参数
    parser.add_argument("--mode", default="sentence", type=str, choices=["sentence", "bag"],
                        help="训练模式: 句子级或包级")
    parser.add_argument("--bag_mode", default="one", type=str, choices=["one", "att"],
                        help="包处理模式: one表示每个包选一个句子，att表示使用注意力机制")
    parser.add_argument("--encoder_type", default="standard", type=str, choices=["standard", "attention"],
                        help="编码器类型")
    parser.add_argument("--multi_label", action="store_true",
                        help="是否是多标签分类")
    
    # 数据相关参数
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="训练数据文件路径")
    parser.add_argument("--dev_file", default=None, type=str, required=True,
                        help="验证数据文件路径")
    parser.add_argument("--test_file", default=None, type=str,
                        help="测试数据文件路径，如果提供会在训练后进行测试")
    parser.add_argument("--relation_file", default=None, type=str, required=True,
                        help="关系标签文件路径")
    
    # 模型相关参数
    parser.add_argument("--model_name_or_path", default="bert-base-chinese", type=str,
                        help="预训练模型名称或路径")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="最大序列长度")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="Dropout比率")
    
    # 训练相关参数
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="模型输出目录")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="批次大小")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="权重衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Adam epsilon参数")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="梯度裁剪阈值")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="训练轮数")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="热身步数")
    parser.add_argument("--logging_steps", default=50, type=int,
                        help="日志记录步数")
    parser.add_argument("--save_steps", default=100, type=int,
                        help="模型保存步数")
    parser.add_argument("--seed", default=42, type=int,
                        help="随机种子")
    parser.add_argument("--no_cuda", action="store_true",
                        help="不使用CUDA")
    
    args = parser.parse_args()
    
    # 训练模型
    train(args)
    
    # 如果提供了测试集，在训练后进行测试
    if args.test_file:
        logger.info("***** 开始测试评估 *****")
        # 加载模型
        if args.mode == "sentence":
            if args.encoder_type == "standard":
                encoder = ChineseBertEncoder(
                    pretrain_path=args.model_name_or_path,
                    max_length=args.max_seq_length
                )
            elif args.encoder_type == "attention":
                encoder = ChineseBertEntityAttentionEncoder(
                    pretrain_path=args.model_name_or_path,
                    max_length=args.max_seq_length
                )
            else:
                raise ValueError(f"不支持的编码器类型: {args.encoder_type}")
            
            # 创建模型
            if args.multi_label:
                model = MultiLabelREModel(
                    encoder=encoder,
                    num_relations=len(load_relations(args.relation_file)),
                    dropout=0.1
                )
            else:
                model = SentenceLevelModel(
                    encoder=encoder,
                    num_relations=len(load_relations(args.relation_file)),
                    dropout=0.1
                )
        
        elif args.mode == "bag":
            # 包级编码器
            encoder = ChineseBagBertEncoder(
                pretrain_path=args.model_name_or_path,
                max_length=args.max_seq_length
            )
            
            # 包级模型
            model = BagAttentionModel(
                encoder=encoder,
                num_relations=len(load_relations(args.relation_file)),
                dropout=0.1
            )
        
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        best_model_path = os.path.join(args.output_dir, "best_model", "model.pt")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        
        # 加载测试数据
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        if args.mode == "sentence":
            test_dataset = ChineseREDataset(
                data_path=args.test_file,
                relation_path=args.relation_file,
                tokenizer=tokenizer,
                max_length=args.max_seq_length
            )
            test_dataloader = get_data_loader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
        elif args.mode == "bag":
            test_dataset = BagREDataset(
                data_path=args.test_file,
                relation_path=args.relation_file,
                tokenizer=tokenizer,
                max_length=args.max_seq_length,
                mode=args.bag_mode
            )
            
            # 使用包级数据的专用collate_fn
            def bag_collate_fn(batch):
                if args.bag_mode == "one":
                    # 单句模式，与普通句子级相同
                    return collate_fn(batch)
                elif args.bag_mode == "att":
                    # 注意力模式，需要特殊处理
                    max_sent_num = max([item['sentence_count'] for item in batch])
                    input_ids = torch.zeros((len(batch), max_sent_num, args.max_seq_length), dtype=torch.long)
                    attention_mask = torch.zeros((len(batch), max_sent_num, args.max_seq_length), dtype=torch.long)
                    entity_mask = torch.zeros((len(batch), max_sent_num, args.max_seq_length), dtype=torch.long)
                    sentence_count = torch.tensor([item['sentence_count'] for item in batch], dtype=torch.long)
                    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
                    
                    for i, item in enumerate(batch):
                        for j in range(item['sentence_count']):
                            input_ids[i, j] = torch.tensor(item['input_ids'][j], dtype=torch.long)
                            attention_mask[i, j] = torch.tensor(item['attention_mask'][j], dtype=torch.long)
                            entity_mask[i, j] = torch.tensor(item['entity_mask'][j], dtype=torch.long)
                    
                    return {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'entity_mask': entity_mask,
                        'sentence_count': sentence_count,
                        'labels': labels
                    }
            
            test_dataloader = get_data_loader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=bag_collate_fn
            )
        
        # 在测试集上评估
        test_results = evaluate(args, model, test_dataloader, device)
        logger.info("***** 测试结果 *****")
        for key, value in test_results.items():
            logger.info(f"  {key} = {value}")


if __name__ == "__main__":
    main() 