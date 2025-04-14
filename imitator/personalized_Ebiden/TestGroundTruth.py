import numpy as np
import trimesh

# === 路径 ===
template_path = 'template_latest_v1.obj'
disp_path = 'disps_segmented_latest_v1/biden_00000.npy'
output_path = 'reconstructed_frame_00000.obj'

# === 加载 mesh 和 displacement ===
template_mesh = trimesh.load(template_path, process=False)
V_template = template_mesh.vertices
F_template = template_mesh.faces
D = np.load(disp_path)

# === 只取一帧，比如第 0 帧 ===
if D.ndim == 3:
    D = D[0]  # shape: (5023, 3)

# === 位移相加 ===
V_i = V_template + D

# === 构建新的 mesh 并保存 ===
reconstructed_mesh = trimesh.Trimesh(vertices=V_i, faces=F_template, process=False)
reconstructed_mesh.export(output_path)

print("✅ Done: Mesh reconstructed and saved as", output_path)
