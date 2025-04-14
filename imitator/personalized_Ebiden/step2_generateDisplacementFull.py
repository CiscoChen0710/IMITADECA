
import os
import numpy as np
from tqdm import tqdm
from scipy.spatial import procrustes

# === 手动读取 .obj 顶点 ===
def load_obj_vertices(filepath):
    vertices = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) == 4:
                    x, y, z = map(float, parts[1:])
                    vertices.append([x, y, z])
    return np.array(vertices)

# === 配置路径 ===
template_path = "template_fixed_5023_faces.obj"
displacement_dir = r"E:\DECA\frames_biden\results"
output_dir = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\disps_410"
os.makedirs(output_dir, exist_ok=True)

# === 加载模板顶点 ===
template_vertices = load_obj_vertices(template_path)
print(f"✅ 模板顶点 shape: {template_vertices.shape}")

# === 遍历 .obj 帧文件 ===
frame_folders = sorted(f for f in os.listdir(displacement_dir) if f.startswith("frame_"))
frame_files = [os.path.join(f, f + ".obj") for f in frame_folders if os.path.isfile(os.path.join(displacement_dir, f, f + ".obj"))]

print(f"📂 共找到 {len(frame_files)} 个 .obj 帧")

success = 0
for rel_path in tqdm(frame_files):
    full_path = os.path.join(displacement_dir, rel_path)
    fname = os.path.basename(rel_path)
    try:
        all_vertices = load_obj_vertices(full_path)

        if all_vertices.shape != (5023, 3):
            print(f"[跳过] {fname}: 顶点 shape 不一致 {all_vertices.shape}")
            continue

        # 姿态对齐
        aligned = all_vertices

        # displacement 不置零任何区域（全脸）
        disp = aligned - template_vertices

        # 保存
        out_name = fname.replace(".obj", ".npy")
        np.save(os.path.join(output_dir, out_name), disp)
        success += 1

    except Exception as e:
        print(f"[错误] {fname}: {e}")

print(f"✅ 成功生成 {success}/{len(frame_files)} 个全脸 displacement 文件，保存在：{output_dir}")