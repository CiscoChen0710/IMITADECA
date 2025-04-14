
import os
import numpy as np

# 配置路径
template_path = "template_fixed_5023_faces.obj"
displacement_dir = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\disps_full_nose_eyebrow"

# 加载模板顶点
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

template_vertices = load_obj_vertices(template_path)
print(f"✅ 模板顶点: {template_vertices.shape}")

# 批量验证 displacement 文件
files = sorted([f for f in os.listdir(displacement_dir) if f.endswith(".npy")])
print(f"📂 共发现 {len(files)} 个 displacement 文件")

error_count = 0
for fname in files:
    path = os.path.join(displacement_dir, fname)
    disp = np.load(path)

    if disp.shape != (5023, 3):
        print(f"[❌] {fname} - 错误 shape: {disp.shape}")
        error_count += 1
        continue

    # 加回模板并检查数值
    restored = template_vertices + disp
    if not np.isfinite(restored).all():
        print(f"[❌] {fname} - 存在非数值顶点")
        error_count += 1

print(f"✅ 验证完成，异常文件数: {error_count} / {len(files)}")