import os
import numpy as np
import trimesh
from tqdm import tqdm

def remove_duplicate_vertices(vertices: np.ndarray, decimals=5):
    """
    去除重复顶点，保留唯一的坐标点
    """
    rounded = np.round(vertices, decimals=decimals)
    unique_vertices, indices = np.unique(rounded, axis=0, return_index=True)
    sorted_indices = np.sort(indices)
    return vertices[sorted_indices]

# === 路径配置 ===
deca_output_root = r"E:\DECA\frames_biden\results"
template_path = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\template.obj"
output_npy_dir = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\disps_unique"

os.makedirs(output_npy_dir, exist_ok=True)

# === 读取并处理 template ===
print(f"📌 Loading template: {template_path}")
template_mesh = trimesh.load(template_path, process=False)
template_vertices_raw = template_mesh.vertices
template_vertices = remove_duplicate_vertices(template_vertices_raw)
print(f"🔢 Template vertices: raw={template_vertices_raw.shape[0]}, unique={template_vertices.shape[0]}")

# === 遍历帧文件夹 ===
frame_folders = sorted(f for f in os.listdir(deca_output_root) if f.startswith("frame_"))
print(f"\n🔍 Found {len(frame_folders)} frame folders.\n")

success_count = 0

for idx, folder in tqdm(enumerate(frame_folders), total=len(frame_folders)):
    obj_filename = f"{folder}.obj"
    mesh_path = os.path.join(deca_output_root, folder, obj_filename)

    if not os.path.exists(mesh_path):
        print(f"[Skip] {mesh_path} not found.")
        continue

    mesh = trimesh.load(mesh_path, process=False)
    frame_vertices_raw = mesh.vertices
    frame_vertices = remove_duplicate_vertices(frame_vertices_raw)

    print(f"[Frame {idx:05d}] Raw: {frame_vertices_raw.shape[0]}, Unique: {frame_vertices.shape[0]}")

    if frame_vertices.shape != template_vertices.shape:
        print(f"[Skip] Vertex count mismatch: {frame_vertices.shape} vs {template_vertices.shape}")
        continue

    disp = frame_vertices - template_vertices
    out_path = os.path.join(output_npy_dir, f"biden_{idx:05d}.npy")
    np.save(out_path, disp)
    success_count += 1

print(f"\n✅ Done! Saved {success_count}/{len(frame_folders)} valid displacement .npy files to:\n{output_npy_dir}")
