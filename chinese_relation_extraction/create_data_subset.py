import os
import json
import glob

def create_subset(input_file, output_file, max_samples):
    """创建包含前max_samples行的数据子集"""
    if not os.path.exists(input_file):
        print(f"警告: 文件 {input_file} 不存在，跳过")
        return False
        
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = []
        for i, line in enumerate(f_in):
            if i >= max_samples:
                break
            lines.append(line)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.writelines(lines)
    print(f"已创建数据子集 {output_file}，包含 {len(lines)} 个样本")
    return True

def update_config_files(config_dir, original_to_small_map):
    """更新配置文件中的数据路径"""
    config_files = glob.glob(os.path.join(config_dir, "*.json"))
    
    for config_file in config_files:
        try:
            # 读取配置文件
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            modified = False
            
            # 更新训练和开发数据路径
            for key in ["train_file", "dev_file"]:
                if key in config:
                    file_path = config[key]
                    for orig, small in original_to_small_map.items():
                        if orig in file_path:
                            config[key] = file_path.replace(orig, small)
                            modified = True
                            print(f"在 {config_file} 中: {orig} -> {small}")
                            break
            
            # 保存修改后的配置
            if modified:
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                print(f"已更新配置文件: {config_file}")
            else:
                print(f"配置文件未修改: {config_file}")
                
        except Exception as e:
            print(f"更新配置文件 {config_file} 时出错: {e}")

def main():
    # 设置样本数量
    max_train_samples = 30000  # 训练集样本数
    max_dev_samples = 10000    # 开发集样本数
    
    # 数据和配置目录
    data_dir = "./data"
    config_dir = "./configs"
    
    # 文件名映射关系 - 基于项目中实际的文件名
    original_to_small_map = {
        "train_sample.txt": "train_sample_small.txt",
        "dev_sample.txt": "dev_sample_small.txt",
        "bag_train_sample.txt": "bag_train_sample_small.txt",
        "bag_dev_sample.txt": "bag_dev_sample_small.txt"
    }
    
    # 创建句子级数据子集
    create_subset(os.path.join(data_dir, "train_sample.txt"), 
                 os.path.join(data_dir, "train_sample_small.txt"), 
                 max_train_samples)
    
    create_subset(os.path.join(data_dir, "dev_sample.txt"), 
                 os.path.join(data_dir, "dev_sample_small.txt"), 
                 max_dev_samples)
    
    # 创建包级数据子集
    create_subset(os.path.join(data_dir, "bag_train_sample.txt"), 
                 os.path.join(data_dir, "bag_train_sample_small.txt"), 
                 max_train_samples)
    
    create_subset(os.path.join(data_dir, "bag_dev_sample.txt"), 
                 os.path.join(data_dir, "bag_dev_sample_small.txt"), 
                 max_dev_samples)
    
    # 查找其他可能的包级文件
    for file in os.listdir(data_dir):
        # 如果有其他包含'bag'的训练或开发文件但不在我们的映射中
        if 'bag' in file and ('train' in file or 'dev' in file) and file not in original_to_small_map:
            is_train = 'train' in file
            small_name = file.replace(".txt", "_small.txt")
            if ".txt" not in file:
                small_name = f"{file}_small"
                
            # 创建子集
            max_samples = max_train_samples if is_train else max_dev_samples
            if create_subset(os.path.join(data_dir, file), 
                            os.path.join(data_dir, small_name), 
                            max_samples):
                # 添加到映射中用于配置文件更新
                original_to_small_map[file] = small_name
    
    print("所有数据子集创建完成")
    
    # 更新配置文件
    if os.path.exists(config_dir):
        update_config_files(config_dir, original_to_small_map)
    else:
        print(f"警告: 配置目录 {config_dir} 不存在，跳过配置文件更新")
    
    print("\n数据子集创建和配置更新已完成！")
    print("现在您可以使用这些子集文件进行训练了。")

if __name__ == "__main__":
    main()
