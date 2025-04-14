import numpy as np
import trimesh
import os

# === 输入路径（选定一帧标准的 .obj）===
source_obj_path = r"E:\DECA\frames_biden\results\frame_00117\frame_00117.obj"
output_template_path = "template_fixed_5023_faces.obj"
output_index_path = "fixed_indices_5023.npy"

# === 加载 mesh ===
print("📌 加载 source mesh...")
mesh = trimesh.load(source_obj_path, process=False)
vertices = mesh.vertices
faces = mesh.faces

# === 提取稳定顺序的 5023 顶点 ===
rounded = np.round(vertices, decimals=5)
_, unique_indices = np.unique(rounded, axis=0, return_index=True)
sorted_unique_indices = np.sort(unique_indices[:5023])

# === 建立索引映射 old_index ➜ new_index ===
old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_unique_indices)}

# === 重建 faces ===
valid_faces = []
for face in faces:
    if all(idx in old_to_new for idx in face):
        new_face = [old_to_new[idx] for idx in face]
        valid_faces.append(new_face)

valid_faces = np.array(valid_faces)
print(f"✅ 原始面数: {len(faces)}, 合法保留面数: {len(valid_faces)}")

# === 构建新 mesh ===
template_vertices = vertices[sorted_unique_indices]
template_mesh = trimesh.Trimesh(vertices=template_vertices, faces=valid_faces, process=False)

# === 导出结果 ===
template_mesh.export(output_template_path)
np.save(output_index_path, sorted_unique_indices)

print(f"\n✅ 模板 mesh 导出为: {output_template_path}")
print(f"🧠 顶点索引保存为: {output_index_path}")
