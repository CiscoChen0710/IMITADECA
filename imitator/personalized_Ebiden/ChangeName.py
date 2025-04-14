import os

# 你的 .npy 文件目录
npy_dir = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\disps_segmented"

for filename in os.listdir(npy_dir):
    if filename.endswith(".npy") and filename.startswith("0"):  # 避免重复改名
        new_name = f"biden_{filename}"
        os.rename(os.path.join(npy_dir, filename), os.path.join(npy_dir, new_name))

print("✅ Renamed all XXXXX.npy files to biden_XXXXX.npy")
