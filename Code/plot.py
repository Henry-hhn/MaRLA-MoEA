# -*-coding:utf-8-*-
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 假设的权重组合和对应的 Q 值
weights = [
    [0.5, 0.2, 0.3],
    [0.6, 0.2, 0.2],
    [0.4, 0.3, 0.3],
    [0.7, 0.15, 0.15],
    [0.3, 0.4, 0.3],
    [0.2, 0.5, 0.3],
    [0.1, 0.6, 0.3],
    [0.8, 0.1, 0.1],
    [0.2, 0.2, 0.6]
]

# 对应的 Q 值
Q_values = [
    0.4555, 0.4400, 0.4350, 0.4200, 0.4150, 0.4050, 0.3900, 0.3800, 0.3700
]

# 将权重组合转换为字符串，用于热力图的标签
weight_labels = [f"{w1:.1f}, {w2:.1f}, {w3:.1f}" for w1, w2, w3 in weights]

# 将 Q 值转换为二维数组（矩阵），这里假设只有一列
Q_matrix = np.array(Q_values).reshape(-1, 1)
# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(Q_matrix, annot=True, fmt=".4f", cmap="viridis", yticklabels=weight_labels, xticklabels=["Q Value"])
plt.title("Heatmap of Q Values for Different Weight Combinations")
plt.ylabel("Weight Combinations (w1, w2, w3)")
plt.xlabel("Performance Metric")
plt.show()
