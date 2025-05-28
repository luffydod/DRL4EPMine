import os
import pickle
import numpy as np
from collections import Counter
from datetime import datetime

def load_expert_data_files(directory):
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    data_info = []
    
    for filename in pkl_files:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # 提取 observation 的 shape 分布
            obs_shapes = [np.array(d['observation']).shape for d in data]
            shape_counter = Counter(obs_shapes)
            data_info.append((filename, len(data), shape_counter, data))

        except Exception as e:
            print(f"无法读取文件 {filename}: {e}")
            data_info.append((filename, 0, Counter(), []))
    
    return data_info

def print_data_info(data_info):
    print("\n找到以下专家数据文件：")
    for idx, (filename, count, shape_counter, _) in enumerate(data_info):
        print(f"[{idx}] {filename} - {count} 条专家数据")
        print("    observation 维度分布：")
        for shape, num in shape_counter.items():
            print(f"      shape={shape}  count={num}")

def merge_selected_data(data_info, selected_indices):
    merged_data = []
    for idx in selected_indices:
        merged_data.extend(data_info[idx][3])
    return merged_data

def main():
    expert_data_dir = 'expert_data'

    data_info = load_expert_data_files(expert_data_dir)
    print_data_info(data_info)

    selected = input("请输入要合并的文件序号（用英文逗号分隔，例如 0,2,3）：").strip()
    try:
        selected_indices = [int(i) for i in selected.split(',') if i.strip().isdigit()]
    except ValueError:
        print("输入格式错误！")
        return

    merged_data = merge_selected_data(data_info, selected_indices)

    save_path = 'expert_data/merged_expert_data.pkl'
    
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(merged_data, f)
        print(f"合并后的专家数据已保存到：{save_path}，共 {len(merged_data)} 条数据")
    except Exception as e:
        print(f"保存文件失败: {e}")

if __name__ == "__main__":
    main()
