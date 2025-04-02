import os
import hashlib

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def generate_entity_id(entity_name):
    """
    为实体生成一个唯一ID
    可以使用哈希或其他方法保证唯一性
    """
    # 使用MD5哈希生成实体ID
    hash_obj = hashlib.md5(entity_name.encode('utf-8'))
    # 取前8位作为ID，并加上"Q"前缀
    return "Q" + hash_obj.hexdigest()[:8]

def convert_to_bag_format(input_line, entity_id_map, sentence_id=None):
    """
    将制表符分隔的数据转换为包级格式
    
    输入格式: 文本\t实体1\t实体2\t关系
    输出格式: 实体1ID\t实体2ID\t关系\t句子ID\t文本\t实体1\t实体2
    """
    parts = input_line.strip().split('\t')
    if len(parts) < 4:
        print(f"警告: 无法转换行 '{input_line}', 格式不正确")
        return input_line
    
    text = parts[0]
    entity1 = parts[1]
    entity2 = parts[2]
    relation = parts[3]
    
    # 获取或创建实体ID
    if entity1 not in entity_id_map:
        entity_id_map[entity1] = generate_entity_id(entity1)
    if entity2 not in entity_id_map:
        entity_id_map[entity2] = generate_entity_id(entity2)
    
    entity1_id = entity_id_map[entity1]
    entity2_id = entity_id_map[entity2]
    
    # 如果没有提供句子ID，默认使用S1
    sid = sentence_id if sentence_id else "S1"
    
    # 转换为包级格式
    return f"{entity1_id}\t{entity2_id}\t{relation}\t{sid}\t{text}\t{entity1}\t{entity2}"

def convert_file(input_file, output_file, entity_id_map=None):
    """
    转换整个文件的格式并保存到新文件
    """
    if entity_id_map is None:
        entity_id_map = {}
        
    try:
        # 读取输入文件
        content = read_file(input_file)
        lines = content.strip().split('\n')
        
        # 转换每一行
        converted_lines = []
        for i, line in enumerate(lines):
            # 可选：为每个句子分配唯一ID
            sentence_id = f"S{i+1}"  # 如果需要唯一ID
            # 或保持所有使用S1：sentence_id = "S1"
            converted_lines.append(convert_to_bag_format(line, entity_id_map, sentence_id))
        
        # 写入输出文件
        write_file(output_file, '\n'.join(converted_lines))
        print(f"成功将 {input_file} 转换并保存到 {output_file}")
        
        return entity_id_map
        
    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {e}")
        return entity_id_map

def main():
    data_dir = "./data"  # 数据目录的路径
    
    # 输入文件
    train_sample_path = os.path.join(data_dir, 'train_sample.txt')
    dev_sample_path = os.path.join(data_dir, 'dev_sample.txt')
    
    # 输出文件
    bag_train_sample_path = os.path.join(data_dir, 'bag_train_sample.txt')
    bag_dev_sample_path = os.path.join(data_dir, 'bag_dev_sample.txt')
    
    # 实体ID映射表 (在训练和开发集之间共享以保持一致)
    entity_id_map = {}
    
    # 转换文件
    if os.path.exists(train_sample_path):
        entity_id_map = convert_file(train_sample_path, bag_train_sample_path, entity_id_map)
    else:
        print(f"文件 {train_sample_path} 不存在")
    
    if os.path.exists(dev_sample_path):
        entity_id_map = convert_file(dev_sample_path, bag_dev_sample_path, entity_id_map)
    else:
        print(f"文件 {dev_sample_path} 不存在")
    
    # 可选：保存实体ID映射表以便将来使用
    entity_map_path = os.path.join(data_dir, 'entity_id_map.txt')
    entity_map_content = '\n'.join([f"{entity}\t{eid}" for entity, eid in entity_id_map.items()])
    write_file(entity_map_path, entity_map_content)
    print(f"实体ID映射表已保存到 {entity_map_path}")

if __name__ == "__main__":
    main()
