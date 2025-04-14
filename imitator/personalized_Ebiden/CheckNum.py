import os
import numpy as np

disps_dir = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\disps_segmented"

# 检查前 5 个文件
files = sorted(f for f in os.listdir(disps_dir) if f.endswith(".npy"))[:5]

for fname in files:
    path = os.path.join(disps_dir, fname)
    arr = np.load(path)
    print(f"{fname} -> shape: {arr.shape}")
