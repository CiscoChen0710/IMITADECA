import os
import numpy as np
from tqdm import tqdm

# 配置路径
input_dir = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\disps_latest_v1"
output_dir = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\disps_segmented_latest_v1"
os.makedirs(output_dir, exist_ok=True)

# 读取所有 npy 文件（确保顺序）
files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
print(f"共找到 {len(files)} 个 displacement 文件")

segment_size = 25
num_segments = len(files) // segment_size

for i in range(num_segments):
    segment_files = files[i * segment_size : (i + 1) * segment_size]
    disps = []

    for fname in segment_files:
        path = os.path.join(input_dir, fname)
        disp = np.load(path)  # shape: (5023, 3)
        disps.append(disp)

    disps = np.stack(disps, axis=0)  # shape: (100, 5023, 3)

    out_path = os.path.join(output_dir, f"biden_{i:05d}.npy")
    np.save(out_path, disps)
    print(f"✅ Saved: {out_path}, shape={disps.shape}")

print(f"✅ 所有 {num_segments} 段完成，输出到 {output_dir}")
