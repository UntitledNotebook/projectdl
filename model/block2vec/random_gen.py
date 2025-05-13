import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_random_voxel_sample(size=16, num_block_types=30, empty_prob=0.3):
    """
    生成一个随机的体素样本
    
    参数:
        size: 体素立方体的边长
        num_block_types: 不同方块类型的数量
        empty_prob: 空方块(值为0)的概率
    
    返回:
        包含随机体素数据的字典
    """
    # 生成随机体素数据，0表示空气/空方块
    voxels = np.random.randint(0, num_block_types, (size, size, size))
    
    # 应用空方块概率
    empty_mask = np.random.random(voxels.shape) < empty_prob
    voxels[empty_mask] = 0
    
    # 添加简单的结构特征，例如地面层
    voxels[0:2, :, :] = np.random.randint(1, 3, size=(2, size, size))  # 底部几层作为地面
    
    return {"voxels": voxels}

def generate_dataset(num_samples=100, save_path=None):
    """
    生成一个完整的随机体素数据集，并保存为 NumPy 文件
    
    参数:
        num_samples: 要生成的样本数量
        save_path: 保存数据集的路径（文件名，例如 "dataset.npy"），可选
    
    返回:
        样本列表，每个样本是一个包含 'voxels' 键的字典
    """
    # 生成样本
    samples = [generate_random_voxel_sample() for _ in range(num_samples)]
    
    # 保存数据集为 NumPy 文件
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.save(save_path, samples)
        print(f"数据集已保存至 {save_path}")
    
    return samples

def visualize_sample(sample, fig_size=(10, 10)):
    """
    可视化一个体素样本
    
    参数:
        sample: 包含'voxels'的字典
        fig_size: 图像尺寸
    """
    voxels = sample['voxels']
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建非空方块的 mask
    mask = voxels > 0
    # 使用颜色映射，归一化颜色信息
    colors = plt.cm.viridis(voxels / voxels.max())
    
    ax.voxels(mask, facecolors=colors[mask], edgecolor='k', alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Voxel Data Visualization')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 生成并保存数据集
    output_file = "random_voxel_dataset.npy"
    dataset = generate_dataset(num_samples=500, save_path=output_file)
    
    # 可视化一个样本
    try:
        sample = dataset[0]
        print(f"样本体素数组形状: {sample['voxels'].shape}")
        print(f"不同方块类型: {np.unique(sample['voxels'])}")
        visualize_sample(sample)
    except Exception as e:
        print(f"可视化失败: {e}")
        
    print("数据集统计:")
    print(f"样本数: {len(dataset)}")
    
    # 如何加载数据集示例：
    print("\n加载数据集示例代码:")
    print("dataset = np.load('random_voxel_dataset.npy', allow_pickle=True)")