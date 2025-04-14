import os
import numpy as np

# ✅ 用 disps_unique 作为输入目录
input_dir = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\disps_unique"
output_dir = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\disps_segmented"
os.makedirs(output_dir, exist_ok=True)

files = sorted(f for f in os.listdir(input_dir) if f.endswith(".npy"))
assert len(files) == 600, "Expected 600 npy files"

for i in range(60):
    group = []
    for j in range(10):
        idx = i * 10 + j
        path = os.path.join(input_dir, files[idx])
        arr = np.load(path)
        assert arr.shape[0] == 5023, f"Unexpected vertex count in {files[idx]}: {arr.shape}"
        group.append(arr)

    segment = np.stack(group, axis=0)  # shape: (10, 5023, 3)
    np.save(os.path.join(output_dir, f"biden_{i:05d}.npy"), segment)

print("✅ All 60 segments saved.")
