import os
import numpy as np
import torch
from tqdm import tqdm
from decalib.models.FLAME import FLAME
from decalib.utils.config import cfg as deca_cfg

# ==== 路径配置 ====
template_obj_path = r"E:\DECA\data\FaceTalk_170731_00024_TA.obj"
npz_root = r"E:\DECA\TestSamples\examples\results"
output_root = r"E:\DECA\TestSamples\examples\reconstructed_objs"

os.makedirs(output_root, exist_ok=True)

# ==== 加载 face indices ====
faces = []
with open(template_obj_path, 'r') as f:
    for line in f:
        if line.startswith('f '):
            parts = line.strip().split()
            face = [int(p.split('/')[0]) for p in parts[1:]]
            faces.append(face)

# ==== 初始化 FLAME 模型 ====
flame = FLAME(deca_cfg.model).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flame = flame.to(device)

# ==== 遍历所有 frame_xxxxx 文件夹 ====
for i in tqdm(range(601)):
    frame_name = f"frame_{i:05d}"
    npz_path = os.path.join(npz_root, frame_name, "flame_params.npz")
    out_obj_path = os.path.join(output_root, f"{frame_name}.obj")

    if not os.path.exists(npz_path):
        print(f"⚠️ Missing: {npz_path}")
        continue

    # === 读取参数 ===
    # data = np.load(npz_path)
    #exp = torch.tensor(data["exp"], dtype=torch.float32).to(device)
    #pose = torch.tensor(data["pose"], dtype=torch.float32).to(device)
    #shape = torch.zeros((1, 100), dtype=torch.float32).to(device)  # 不使用 shape，设为 0
    # shape = torch.zeros((1, 100), dtype=torch.float32).to(device)
    # pose = torch.zeros((1, 6), dtype=torch.float32).to(device)
    # exp = torch.zeros((1, 50), dtype=torch.float32).to(device)  # FLAME 的 exp 通常是 50 维
    # === 读取参数 ===
    data = np.load(npz_path)
    exp = torch.tensor(data["exp"], dtype=torch.float32).to(device)

    # 原始 pose 可能是 6维或 15x3=45维，先加载原始 pose：
    orig_pose = data["pose"]
    pose = torch.tensor(orig_pose, dtype=torch.float32).to(device)

    # 🚫 禁用前 3 维 global rotation（头部朝向）
    pose[:, 0:3] = 0.0  # 只保留 jaw rotation（通常在 6:9）

    # ✅ 禁用 shape（保持脸型一致）
    shape = torch.zeros((1, 100), dtype=torch.float32).to(device)



    # === 获取顶点 ===
    with torch.no_grad():
        verts, _, _ = flame(shape, exp, pose)
        verts = verts[0].cpu().numpy()

    # === 写入 obj ===
    with open(out_obj_path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

print("✅ 所有重构完成！")
