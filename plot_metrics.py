#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator

def load_csv_file(file_path):
    """加载CSV文件并返回DataFrame"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"加载文件 {file_path} 失败: {e}")
        return None

def smooth_data(data, weight=0.85):
    """使用指数移动平均对数据进行平滑处理"""
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_metrics(data_files, labels, data_type, output_file=None, smooth_factor=0.85):
    """绘制指标曲线"""
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    colors = sns.color_palette("husl", len(data_files))
    
    for i, (file_path, label) in enumerate(zip(data_files, labels)):
        df = load_csv_file(file_path)
        if df is None:
            continue
            
        x = df['Step'].values
        y = df['Value'].values
        
        # 绘制原始数据（透明）
        plt.plot(x, y, alpha=0.3, color=colors[i])
        
        # 绘制平滑后的数据
        smoothed_y = smooth_data(y, smooth_factor)
        plt.plot(x, smoothed_y, label=label, linewidth=2.5, color=colors[i])
    
    # 设置图表属性
    plt.title(f'training process {data_type} comparison', fontsize=16, fontweight='bold')
    plt.xlabel('training steps', fontsize=14)
    
    if data_type == 'ep_len_mean':
        plt.ylabel('episode mean length', fontsize=14)
    elif data_type == 'ep_rew_mean':
        plt.ylabel('episode mean reward', fontsize=14)
    else:
        plt.ylabel('value', fontsize=14)
    
    plt.legend(fontsize=12, loc='best', frameon=True, fancybox=True, framealpha=0.8, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置x轴为整数刻度
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 添加水印
    plt.figtext(0.99, 0.01, 'DRL_EP', fontsize=10, color='gray', ha='right', alpha=0.5)
    
    # 美化
    sns.despine()
    
    # 保存或显示
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图表已保存至 {output_file}")
    else:
        plt.tight_layout()
        plt.show()

def list_csv_files(directory):
    """列出目录中的所有CSV文件"""
    if not os.path.exists(directory):
        print(f"错误: 目录 '{directory}' 不存在")
        return []
        
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    return files

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练指标绘图工具')
    parser.add_argument('--prefix', type=str, default='metrics_data', help='文件前缀')
    parser.add_argument('--type', type=str, choices=['ep_len_mean', 'ep_rew_mean'], 
                        required=True, help='要绘制的数据类型')
    parser.add_argument('--output', type=str, help='输出图片文件路径（可选）')
    parser.add_argument('--smooth', type=float, default=0.50, help='平滑因子 (0-1之间，默认0.50)')
    
    args = parser.parse_args()
    
    # 确定数据目录
    data_dir = args.prefix + '/' + args.type
    
    # 列出可用的CSV文件
    csv_files = list_csv_files(data_dir)
    if not csv_files:
        print(f"在 {data_dir} 目录中没有找到CSV文件")
        return
    
    # 显示文件列表供用户选择
    print(f"\n在 {data_dir} 目录中找到以下CSV文件:")
    for i, file in enumerate(csv_files):
        print(f"[{i}] {file}")
    
    # 用户选择文件
    try:
        selection = input("\n请输入要绘制的文件序号（用逗号分隔多个序号）: ")
        selected_indices = [int(idx.strip()) for idx in selection.split(',')]
        
        # 验证选择的有效性
        if any(idx < 0 or idx >= len(csv_files) for idx in selected_indices):
            print("错误: 无效的文件序号")
            return
            
        selected_files = [os.path.join(data_dir, csv_files[idx]) for idx in selected_indices]
        
        # 用户提供标签
        print("\n请为选择的文件提供标签:")
        labels = []
        for i, file in enumerate([csv_files[idx] for idx in selected_indices]):
            default_label = os.path.splitext(file)[0]
            label = input(f"文件 '{file}' 的标签 [默认: {default_label}]: ")
            labels.append(label if label else default_label)
        
        # 绘制图表
        plot_metrics(selected_files, labels, args.type, args.output, args.smooth)
        
    except ValueError:
        print("错误: 请输入有效的数字序号")
    except KeyboardInterrupt:
        print("\n操作已取消")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()