# coding=utf-8
"""
简化的训练启动脚本
"""

import os
import sys
import json
import argparse
from train import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="配置文件路径")
    parser.add_argument("--train_file", required=True, type=str, help="训练数据文件路径")
    parser.add_argument("--dev_file", required=True, type=str, help="验证数据文件路径")
    parser.add_argument("--relation_file", required=True, type=str, help="关系标签文件路径")
    parser.add_argument("--output_dir", required=True, type=str, help="模型输出目录")
    
    args = parser.parse_args()
    
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建完整的参数对象
    class Args:
        def __init__(self, config, cmd_args):
            # 设置命令行指定的参数
            self.train_file = cmd_args.train_file
            self.dev_file = cmd_args.dev_file
            self.relation_file = cmd_args.relation_file
            self.output_dir = cmd_args.output_dir
            
            # 设置配置文件中的参数
            for key, value in config.items():
                setattr(self, key, value)
            
            # 设置默认值
            self.no_cuda = False  # 添加no_cuda属性，默认为False
    
    # 创建参数对象
    train_args = Args(config, args)
    
    # 打印参数
    print("训练参数:")
    for arg in vars(train_args):
        print(f"  {arg}: {getattr(train_args, arg)}")
    
    # 开始训练
    train(train_args)

if __name__ == "__main__":
    main() 