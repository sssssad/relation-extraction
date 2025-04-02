# coding=utf-8
"""
使用配置文件训练中文关系抽取模型的脚本
"""

import os
import sys
import json
import argparse
import logging

# 设置日志
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_file):
    """加载配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def config_to_args(config):
    """将配置转换为命令行参数"""
    args = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.append(f"--{key}")
            args.append(str(value))
    return args


def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="配置文件路径")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 打印配置信息
    logger.info(f"使用配置文件: {args.config}")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # 将配置转换为命令行参数
    cmd_args = config_to_args(config)
    cmd = "python train.py " + " ".join(cmd_args)
    
    # 打印命令
    logger.info(f"即将执行命令: {cmd}")
    
    # 执行命令
    os.system(cmd)


if __name__ == "__main__":
    main() 