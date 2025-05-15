import numpy as np
import os

def transform_dataset(input_file, output_file, target_dtype=np.int64):
    """
    加载原始数据集，对整个数组进行类型转换，
    并保存转换后的数据集到 output_file。

    参数:
        input_file: 原始 npy 数据集文件路径，数据形状为 (n, 16, 16, 16)
        output_file: 转换后数据集保存的路径
        target_dtype: 目标数据类型（默认 np.int32）
    """
    # 加载原始数据集（非 pickle 模式即可）
    dataset = np.load(input_file)
    print(f"原始数据类型: {dataset.dtype}, 形状: {dataset.shape}")

    # 转换整个数组的数据类型
    transformed = dataset.astype(target_dtype)
    
    # 如果 output_file 包含目录，则确保目录存在
    directory = os.path.dirname(output_file)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
      
    np.save(output_file, transformed)
    print(f"Transformed dataset saved to {output_file}")

if __name__ == "__main__":
    # 示例：将 'subset.npy' 转换后保存为 'subset_transformed.npy'
    transform_dataset("processed_block_ids.npy", "processed_block_ids_transformed.npy")