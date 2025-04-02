import os
import json

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def convert_json_to_tab_format(json_line):
    """将JSON行转换为制表符分隔的文本格式"""
    try:
        data = json.loads(json_line)
        text = data.get('text', '')
        labels = data.get('labels', [])
        # 将文本和所有标签用制表符连接
        return text + '\t' + '\t'.join(labels)
    except json.JSONDecodeError:
        print(f"无法解析JSON行: {json_line}")
        return json_line

def process_files(data_dir):
    # 读取需要转换的文件
    try:
        train_path = os.path.join(data_dir, 'train.txt')
        dev_path = os.path.join(data_dir, 'dev.txt')
        
        train_content = read_file(train_path)
        dev_content = read_file(dev_path)
        
        # 转换每一行
        train_lines = train_content.strip().split('\n')
        dev_lines = dev_content.strip().split('\n')
        
        # 将JSON行转换为制表符分隔的格式
        new_train_lines = [convert_json_to_tab_format(line) for line in train_lines]
        new_dev_lines = [convert_json_to_tab_format(line) for line in dev_lines]
        
        # 写入转换后的文件
        write_file(train_path, '\n'.join(new_train_lines))
        write_file(dev_path, '\n'.join(new_dev_lines))
        
        # 处理bag文件
        bag_files = [f for f in os.listdir(data_dir) if f.startswith('bag')]
        
        # 查找train和dev对应的bag文件
        train_bag_file = next((f for f in bag_files if 'train' in f.lower()), None)
        dev_bag_file = next((f for f in bag_files if 'dev' in f.lower()), None)
        
        # 读取sample文件以获取bag格式
        train_sample_path = os.path.join(data_dir, 'train_sample.txt')
        dev_sample_path = os.path.join(data_dir, 'dev_sample.txt')
        
        if os.path.exists(train_sample_path) and train_bag_file:
            train_sample = read_file(train_sample_path)
            train_bag_path = os.path.join(data_dir, train_bag_file)
            update_bag_file(train_bag_path, train_sample_path)
            
        if os.path.exists(dev_sample_path) and dev_bag_file:
            dev_sample = read_file(dev_sample_path)
            dev_bag_path = os.path.join(data_dir, dev_bag_file)
            update_bag_file(dev_bag_path, dev_sample_path)
            
        print("转换完成!")
        
    except Exception as e:
        print(f"处理文件时出错: {e}")

def update_bag_file(bag_file_path, sample_file_path):
    """根据sample文件更新bag文件的内容"""
    try:
        # 读取sample文件的第一行来确定其格式
        sample_content = read_file(sample_file_path)
        sample_lines = sample_content.strip().split('\n')
        
        if not sample_lines:
            print(f"样本文件 {sample_file_path} 为空")
            return
            
        # 读取bag文件
        bag_content = read_file(bag_file_path)
        
        # 如果bag文件是JSON格式，我们需要提取并修改它
        try:
            # 尝试解析sample的第一行，查看是否为JSON
            sample_first_line = json.loads(sample_lines[0])
            is_sample_json = True
        except json.JSONDecodeError:
            # 如果不是JSON，说明sample已经是制表符分隔的格式
            is_sample_json = False
            
        # 尝试解析整个bag文件
        try:
            bag_data = json.loads(bag_content)
            # 如果bag文件是JSON，我们需要更新其内容
            write_file(bag_file_path, bag_content)  # 先不做修改，仅保存文件
        except json.JSONDecodeError:
            # 如果bag文件不是JSON格式，直接替换内容
            bag_content = sample_content
            write_file(bag_file_path, bag_content)
            
    except Exception as e:
        print(f"更新bag文件 {bag_file_path} 时出错: {e}")

if __name__ == "__main__":
    data_dir = "./data"  # 数据目录的路径
    process_files(data_dir)
