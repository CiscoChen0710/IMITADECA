
import os
import numpy as np

segment_dir = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\disps_segmented_full"

files = sorted([f for f in os.listdir(segment_dir) if f.endswith(".npy")])
print(f"📂 找到分段 displacement 文件: {len(files)}")

if len(files) != 60:
    print(f"[❌] 文件数不是 60，而是 {len(files)}")
else:
    print("✅ 段数正确: 60")

error_count = 0
for fname in files:
    path = os.path.join(segment_dir, fname)
    try:
        arr = np.load(path)
        if arr.shape != (10, 5023, 3):
            print(f"[❌] {fname} - 错误 shape: {arr.shape}")
            error_count += 1
        elif not np.isfinite(arr).all():
            print(f"[❌] {fname} - 存在 NaN/inf")
            error_count += 1
    except Exception as e:
        print(f"[❌] {fname} - 加载失败: {e}")
        error_count += 1

print(f"✅ 验证完成，异常文件数: {error_count} / {len(files)}")
