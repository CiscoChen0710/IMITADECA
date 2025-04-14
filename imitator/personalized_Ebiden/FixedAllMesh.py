import os
import numpy as np
import trimesh
from tqdm import tqdm

# === 配置路径 ===
fixed_indices_path = "fixed_indices_from_117.npy"
template_obj_path = "template_fixed_with_faces.obj"
deca_output_root = r"E:\DECA\frames_biden\results"
output_npy_dir = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\disps_aligned"

os.makedirs(output_npy_dir, exist_ok=True)

# === 加载 template 和 index ===
print("📌 加载 fixed indices 和 template.obj...")
fixed_indices = np.load(fixed_indices_path)  # shape: (5023,)
template_mesh = trimesh.load(template_obj_path, process=False)
template_vertices = template_mesh.vertices    # (5023, 3)

# === 遍历每帧 DECA 输出 ===
frame_folders = sorted(f for f in os.listdir(deca_output_root) if f.startswith("frame_"))
print(f"🔍 共找到 {len(frame_folders)} 帧")

success = 0
for idx, folder in tqdm(enumerate(frame_folders), total=len(frame_folders)):
    mesh_path = os.path.join(deca_output_root, folder, f"{folder}.obj")
    if not os.path.exists(mesh_path):
        print(f"[Skip] 找不到: {mesh_path}")
        continue

    mesh = trimesh.load(mesh_path, process=False)
    frame_vertices = mesh.vertices

    if frame_vertices.shape[0] <= np.max(fixed_indices):
        print(f"[Skip] 顶点数不足: {frame_vertices.shape[0]}")
        continue

    aligned_vertices = frame_vertices[fixed_indices]  # 按一致顺序提取
    displacement = aligned_vertices - template_vertices  # 差值
    out_path = os.path.join(output_npy_dir, f"biden_{idx:05d}.npy")
    np.save(out_path, displacement)
    success += 1

print(f"\n✅ 已成功生成 {success}/{len(frame_folders)} 个 displacement 文件保存至:\n{output_npy_dir}")
