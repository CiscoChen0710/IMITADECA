import numpy as np
import trimesh

# 路径配置
template_path = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\template_unique.obj"
displacement_path = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\disps_segmented\biden_00000.npy"

# 读取模板 mesh 顶点和面
template_mesh = trimesh.load(template_path, process=False)
vertices_template = template_mesh.vertices  # (5023, 3)
faces = template_mesh.faces

# 读取 displacement（第 0 帧）
displacements = np.load(displacement_path)  # shape: (10, 5023, 3)
displaced_vertices = vertices_template + displacements[0]

# 构建新 mesh
new_mesh = trimesh.Trimesh(vertices=displaced_vertices, faces=faces)

# 显示结果
new_mesh.show()
