import os
import numpy as np
import trimesh
from tqdm import tqdm
from scipy.spatial import KDTree

# 路径配置
deca_result_root = r"E:\DECA\frames_biden\results"
template_path = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_biden\template.obj"
output_dir = r"D:\Purdue\ECE57000\reimplemention\Imitator\warped_DECA_obj"
os.makedirs(output_dir, exist_ok=True)

# 加载模板和 faces
template_mesh = trimesh.load(template_path, process=False)
template_vertices = np.asarray(template_mesh.vertices)
faces = template_mesh.faces  # faces 用 template 的

# 批量处理前 N 帧
frame_names = [f"frame_{i:05d}" for i in range(1, 11)]  # 修改这里控制帧数

for frame_name in tqdm(frame_names, desc="Warping"):
    source_path = os.path.join(deca_result_root, frame_name, f"{frame_name}.obj")
    if not os.path.exists(source_path):
        print(f"[Skip] {source_path} not found")
        continue

    source_mesh = trimesh.load(source_path, process=False)
    source_vertices = np.asarray(source_mesh.vertices)

    # KDTree 最近邻匹配：每个 template 顶点找 source 上最近点
    tree = KDTree(source_vertices)
    dists, indices = tree.query(template_vertices, k=1)
    warped_vertices = source_vertices[indices]

    # 输出 warped obj
    warped_mesh = trimesh.Trimesh(vertices=warped_vertices, faces=faces)
    warped_mesh.export(os.path.join(output_dir, f"{frame_name}.obj"))

print("✅ Warping 完成")
