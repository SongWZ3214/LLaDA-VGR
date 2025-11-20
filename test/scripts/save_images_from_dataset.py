#!/usr/bin/env python3
"""
脚本1: 从parquet数据集读取binary列并保存为图片
编号与detailcaps_outputs.json保持一致
"""

import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import base64
import io
import re
import codecs
import sys

# 添加父目录到路径，以便导入generate_demo中的load_image_from_data
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# 导入图像加载函数
from train.generate_test import load_image_from_data

# 配置参数
DATASET_PATH = "/data2/swz/LLaDA-VGR/test/data/DetailCaps-4870_refined_EN.parquet"
OUTPUT_JSON_PATH = "/data2/swz/LLaDA-VGR/test/result/detailcaps_outputs.json"
OUTPUT_IMAGE_DIR = "/data2/swz/LLaDA-VGR/test/result/images"

def main():
    """主函数"""
    # 创建输出目录
    output_dir = Path(OUTPUT_IMAGE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取JSON文件，获取需要保存的索引列表
    print("正在读取detailcaps_outputs.json...")
    with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        outputs_data = json.load(f)
    
    # 获取所有索引
    indices = [item['index'] for item in outputs_data]
    print(f"找到 {len(indices)} 条数据需要保存图片")
    
    # 读取parquet文件
    print(f"正在读取数据集: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    print(f"数据集大小: {len(df)} 行")
    
    # 检查binary列是否存在
    if 'binary' not in df.columns:
        print("错误: 数据集中没有找到'binary'列")
        print(f"可用列: {list(df.columns)}")
        return
    
    # 保存图片
    success_count = 0
    fail_count = 0
    
    print(f"开始保存图片到: {output_dir}")
    for idx in tqdm(indices, desc="保存图片"):
        try:
            # 检查索引是否有效
            if idx >= len(df):
                print(f"警告: 索引 {idx} 超出数据集范围 (最大索引: {len(df)-1})")
                fail_count += 1
                continue
            
            # 获取binary数据
            binary_data = df.iloc[idx]['binary']
            
            if pd.isna(binary_data):
                print(f"警告: 索引 {idx} 的binary数据为空")
                fail_count += 1
                continue
            
            # 加载图像
            image = load_image_from_data(binary_data)
            
            if image is None:
                print(f"警告: 索引 {idx} 无法加载图像")
                fail_count += 1
                continue
            
            # 保存图片（使用PNG格式以保证质量）
            image_path = output_dir / f"{idx:06d}.png"
            image.save(image_path, 'PNG')
            
            success_count += 1
            
        except Exception as e:
            print(f"错误: 处理索引 {idx} 时出错: {e}")
            fail_count += 1
            continue
    
    # 输出统计信息
    print("\n" + "="*50)
    print("保存完成!")
    print(f"成功: {success_count} 张")
    print(f"失败: {fail_count} 张")
    print(f"图片保存目录: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
