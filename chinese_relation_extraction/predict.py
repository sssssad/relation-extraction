# coding=utf-8
"""
中文关系抽取模型预测脚本
"""

import os
import sys
import argparse
import logging
import json
import torch
from transformers import BertTokenizer
from tqdm import tqdm

from processors.chinese_bert_encoder import ChineseBertEncoder, ChineseBertEntityAttentionEncoder, ChineseBagBertEncoder
from models.chinese_re_model import SentenceLevelModel, BagAttentionModel, MultiLabelREModel
from utils import load_relations, convert_example_to_feature

# 设置日志
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_single_sentence(text, head_entity, tail_entity, model, tokenizer, id2relation, device, max_length=128):
    """对单个句子进行关系预测"""
    
    # 转换为模型输入
    feature = convert_example_to_feature(
        text=text,
        head_entity=head_entity,
        tail_entity=tail_entity,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    if feature is None:
        logger.warning(f"无法在文本中找到实体: '{head_entity}' 或 '{tail_entity}'")
        return None, None
    
    # 将特征转换为张量
    input_ids = torch.tensor([feature['input_ids']], dtype=torch.long).to(device)
    attention_mask = torch.tensor([feature['attention_mask']], dtype=torch.long).to(device)
    entity_mask = torch.tensor([feature['entity_mask']], dtype=torch.long).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        pred_label, probs = model.predict(input_ids, attention_mask, entity_mask)
    
    # 获取预测结果
    pred_label = pred_label.cpu().numpy()[0]
    
    if hasattr(model, 'threshold'):  # 多标签模型
        # 获取所有预测为正的关系
        pred_relations = []
        probs = probs.cpu().numpy()[0]
        for i, prob in enumerate(probs):
            if prob > model.threshold:
                pred_relations.append((id2relation[i], float(prob)))
        return pred_relations, probs
    else:  # 单标签模型
        pred_relation = id2relation[pred_label]
        prob = float(probs.cpu().numpy()[0][pred_label])
        return (pred_relation, prob), probs.cpu().numpy()[0]


def predict_from_file(args):
    """从文件读取数据并预测"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    
    # 加载关系标签
    relation2id = load_relations(args.relation_file)
    id2relation = {v: k for k, v in relation2id.items()}
    num_relations = len(relation2id)
    logger.info(f"关系类别数量: {num_relations}")
    
    # 创建编码器
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
                num_relations=num_relations,
                dropout=0.1
            )
        else:
            model = SentenceLevelModel(
                encoder=encoder,
                num_relations=num_relations,
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
            num_relations=num_relations,
            dropout=0.1
        )
    
    else:
        raise ValueError(f"不支持的模式: {args.mode}")
    
    # 加载模型
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info("***** 开始预测 *****")
    
    # 读取输入文件
    with open(args.input_file, 'r', encoding='utf-8') as f_in, \
         open(args.output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="预测中"):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 3:
                logger.warning(f"输入格式不正确: {line}")
                continue
            
            text, head_entity, tail_entity = parts[:3]
            
            # 预测
            pred_result, probs = predict_single_sentence(
                text=text,
                head_entity=head_entity,
                tail_entity=tail_entity,
                model=model,
                tokenizer=tokenizer,
                id2relation=id2relation,
                device=device,
                max_length=args.max_seq_length
            )
            
            if pred_result is None:
                # 预测失败，写入特殊标记
                f_out.write(f"{text}\t{head_entity}\t{tail_entity}\tNO_RELATION\t0.0\n")
                continue
            
            # 输出预测结果
            if isinstance(pred_result, list):  # 多标签情况
                if not pred_result:  # 没有预测出任何关系
                    f_out.write(f"{text}\t{head_entity}\t{tail_entity}\tNO_RELATION\t0.0\n")
                else:
                    for rel, prob in pred_result:
                        f_out.write(f"{text}\t{head_entity}\t{tail_entity}\t{rel}\t{prob:.4f}\n")
            else:  # 单标签情况
                rel, prob = pred_result
                f_out.write(f"{text}\t{head_entity}\t{tail_entity}\t{rel}\t{prob:.4f}\n")
    
    logger.info(f"预测完成，结果已保存至: {args.output_file}")


def predict_interactive(args):
    """交互式预测"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    
    # 加载关系标签
    relation2id = load_relations(args.relation_file)
    id2relation = {v: k for k, v in relation2id.items()}
    num_relations = len(relation2id)
    logger.info(f"关系类别数量: {num_relations}")
    
    # 创建编码器
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
                num_relations=num_relations,
                dropout=0.1
            )
        else:
            model = SentenceLevelModel(
                encoder=encoder,
                num_relations=num_relations,
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
            num_relations=num_relations,
            dropout=0.1
        )
    
    else:
        raise ValueError(f"不支持的模式: {args.mode}")
    
    # 加载模型
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info("***** 交互式预测 *****")
    logger.info("输入格式: 文本\\t头实体\\t尾实体")
    logger.info("输入'exit'退出")
    
    while True:
        try:
            # 读取用户输入
            user_input = input("\n请输入: ")
            
            if user_input.lower() == 'exit':
                break
            
            parts = user_input.split('\t')
            if len(parts) != 3:
                print("输入格式不正确，请按照 '文本\\t头实体\\t尾实体' 的格式输入")
                continue
            
            text, head_entity, tail_entity = parts
            
            # 预测
            pred_result, probs = predict_single_sentence(
                text=text,
                head_entity=head_entity,
                tail_entity=tail_entity,
                model=model,
                tokenizer=tokenizer,
                id2relation=id2relation,
                device=device,
                max_length=args.max_seq_length
            )
            
            if pred_result is None:
                print(f"无法在文本中找到实体: '{head_entity}' 或 '{tail_entity}'")
                continue
            
            # 输出预测结果
            print("\n预测结果:")
            if isinstance(pred_result, list):  # 多标签情况
                if not pred_result:  # 没有预测出任何关系
                    print("无关系")
                else:
                    for i, (rel, prob) in enumerate(sorted(pred_result, key=lambda x: x[1], reverse=True)):
                        print(f"{i+1}. {rel} (置信度: {prob:.4f})")
            else:  # 单标签情况
                rel, prob = pred_result
                print(f"关系: {rel} (置信度: {prob:.4f})")
                
                # 输出前5个最可能的关系
                top5_indices = torch.argsort(torch.tensor(probs), descending=True)[:5].tolist()
                print("\n其他可能的关系:")
                for i, idx in enumerate(top5_indices[1:], 1):  # 跳过第一个（已经显示过）
                    print(f"{i+1}. {id2relation[idx]} (置信度: {probs[idx]:.4f})")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")
    
    print("退出预测")


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser()
    
    # 基础参数
    parser.add_argument("--mode", default="sentence", type=str, choices=["sentence", "bag"],
                        help="预测模式: 句子级或包级")
    parser.add_argument("--encoder_type", default="standard", type=str, choices=["standard", "attention"],
                        help="编码器类型")
    parser.add_argument("--multi_label", action="store_true",
                        help="是否是多标签分类")
    parser.add_argument("--interactive", action="store_true",
                        help="交互式预测")
    
    # 数据相关参数
    parser.add_argument("--input_file", default=None, type=str,
                        help="输入文件路径，每行格式: 文本\\t头实体\\t尾实体")
    parser.add_argument("--output_file", default=None, type=str,
                        help="输出文件路径，每行格式: 文本\\t头实体\\t尾实体\\t关系\\t置信度")
    parser.add_argument("--relation_file", default=None, type=str, required=True,
                        help="关系标签文件路径")
    
    # 模型相关参数
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="模型文件路径")
    parser.add_argument("--model_name_or_path", default="bert-base-chinese", type=str,
                        help="预训练模型名称或路径")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="最大序列长度")
    parser.add_argument("--no_cuda", action="store_true",
                        help="不使用CUDA")
    
    args = parser.parse_args()
    
    # 参数检查
    if not args.interactive and (args.input_file is None or args.output_file is None):
        parser.error("非交互式模式下必须指定 --input_file 和 --output_file")
    
    # 开始预测
    if args.interactive:
        predict_interactive(args)
    else:
        predict_from_file(args)


if __name__ == "__main__":
    main() 