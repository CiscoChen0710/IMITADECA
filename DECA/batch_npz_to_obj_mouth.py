import os
import numpy as np
import torch
from tqdm import tqdm
from decalib.models.FLAME import FLAME
from decalib.utils.config import cfg as deca_cfg

# ==== 路径配置 ====
template_obj_path = r"E:\DECA\data\FaceTalk_170731_00024_TA.obj"
npz_root = r"E:\DECA\TestSamples\examples\results"
output_root = r"E:\DECA\TestSamples\examples\reconstructed_objs_mouth_only"
mouth_indices_path = r"E:\DECA\data\mouth_indices_5023.npy"  # ✅ 你自己刚刚提取出来的

os.makedirs(output_root, exist_ok=True)

# ==== 加载原始 base mesh ====
print("📌 加载 base template mesh...")
with open(template_obj_path, 'r') as f:
    lines = f.readlines()

base_vertices = np.array([list(map(float, line.strip().split()[1:]))
                         for line in lines if line.startswith('v ')])
faces = [line for line in lines if line.startswith('f ')]

# ==== 加载 mouth 区域索引 ====
mouth_indices = np.load(mouth_indices_path)  # shape: (N_mouth,)

# ==== 初始化 FLAME 模型 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flame = FLAME(deca_cfg.model).eval().to(device)  # ✅ 替换成这行

# ==== 遍历每一帧 ====
for i in tqdm(range(600)):
    frame_name = f"frame_{i:05d}"
    npz_path = os.path.join(npz_root, frame_name, "flame_params.npz")
    out_obj_path = os.path.join(output_root, f"{frame_name}.obj")

    if not os.path.exists(npz_path):
        print(f"⚠️ Missing: {npz_path}")
        continue

    # === 读取表情和姿态参数 ===
    data = np.load(npz_path)
    exp = torch.tensor(data["exp"], dtype=torch.float32).to(device)
    pose = torch.tensor(data["pose"], dtype=torch.float32).to(device)
    shape = torch.zeros((1, 100), dtype=torch.float32).to(device)

    # === 生成新 mesh，并仅更新嘴部区域 ===
    with torch.no_grad():
        verts, _, _ = flame(shape, exp, pose)
        verts = verts[0].cpu().numpy()

    # === 替换掉 base mesh 中的嘴部顶点 ===
    updated_vertices = base_vertices.copy()
    updated_vertices[mouth_indices] = verts[mouth_indices]

    # === 写入新的 obj ===
    with open(out_obj_path, 'w') as f:
        for v in updated_vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(face)  # face 不变，直接写原始的行

print("✅ 所有 mouth-only obj 重建完成！")
