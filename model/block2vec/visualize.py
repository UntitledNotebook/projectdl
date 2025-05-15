import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_embeddings(embedding_pickle="output/block2vec/index_embeddings.pkl",
                         embedding_npy="output/block2vec/embeddings.npy",
                         output_svg_path="output/block2vec/embeddings_visualization.svg"):
    # 加载 pickle 保存的 block index 到嵌入的映射
    with open(embedding_pickle, "rb") as f:
        index_embeddings = pickle.load(f)
    
    # 或者加载完整嵌入矩阵
    embeddings_array = np.load(embedding_npy)
    
    # 这里我们将使用 pickle 加载的映射，排序后得到 indices 与对应的嵌入向量
    sorted_indices = sorted(index_embeddings.keys())
    embeddings = np.array([index_embeddings[i] for i in sorted_indices])
    
    # 使用 PCA 将嵌入降维到 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # 绘制散点图
    plt.figure(figsize=(10,10))
    sc = plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=sorted_indices, cmap='tab10', s=10)
    plt.colorbar(sc, label="Block Index")
    
    # 添加标签信息
    for i, idx in enumerate(sorted_indices):
        plt.annotate(str(idx), (embeddings_2d[i,0], embeddings_2d[i,1]),
                     textcoords="offset points", xytext=(0,3), ha='center' , fontsize=6)
    
    plt.title("Block Embeddings Visualized with PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()

    output_dir = os.path.dirname(output_svg_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    # 保存为 SVG 文件
    plt.savefig(output_svg_path, format='svg', bbox_inches='tight')
    print(f"Plot saved to {output_svg_path}")

    plt.show()

if __name__ == "__main__":
    visualize_embeddings()