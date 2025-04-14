import os
from pathlib import Path

root_dir = "frames_biden/results"

for folder in sorted(Path(root_dir).iterdir()):
    if folder.is_dir():
        # 删除 .mtl 文件
        for mtl_file in folder.glob("*.mtl"):
            mtl_file.unlink()

        # 处理 .obj 文件（移除 mtllib 和 usemtl）
        for obj_file in folder.glob("*.obj"):
            with open(obj_file, 'r') as f:
                lines = f.readlines()

            clean_lines = [line for line in lines if not (line.startswith("mtllib") or line.startswith("usemtl"))]

            with open(obj_file, 'w') as f:
                f.writelines(clean_lines)

print("✅ 清理完成，只保留纯净 .obj 白模")
