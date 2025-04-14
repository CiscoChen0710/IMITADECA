import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === 加载 mesh ===
obj_path = "template_latest.obj"  # 你的 obj 路径
mesh = trimesh.load(obj_path, process=False)
vertices = mesh.vertices  # (5023, 3)

# === 基于经验规则过滤嘴部 ===
x_min, x_max = -0.04, 0.04
y_min, y_max = -0.08, -0.05
z_min, z_max = 0.01, 0.07

mask = (
    (vertices[:, 0] > x_min) & (vertices[:, 0] < x_max) &
    (vertices[:, 1] > y_min) & (vertices[:, 1] < y_max) &
    (vertices[:, 2] > z_min) & (vertices[:, 2] < z_max)
)
mouth_indices = np.where(mask)[0]

# === 保存 ===
np.save("mouth_indices_5023.npy", mouth_indices)
print(f"✅ 提取了 {len(mouth_indices)} 个嘴部顶点并保存为 mouth_indices_5023.npy")

# === 可视化检查 ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, alpha=0.05, label='All Vertices')
ax.scatter(vertices[mouth_indices, 0], vertices[mouth_indices, 1], vertices[mouth_indices, 2],
           color='red', s=10, label='Mouth Vertices')
ax.set_title('Mouth Region Highlighted')
ax.legend()
plt.show()
