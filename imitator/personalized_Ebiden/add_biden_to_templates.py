import trimesh
import pickle

# === 路径设置 ===
template_obj_path = "template_latest_v1.obj"  # 你自己的模板 .obj 路径
original_templates_path = "templates.pkl"  # 原始 VOCA templates.pkl
output_templates_path = "templates_biden.pkl"  # 输出新文件名

# === 加载 .obj 顶点 ===
print("📌 加载你的 template.obj...")
mesh = trimesh.load(template_obj_path, process=False)
vertices = mesh.vertices.astype("float32")  # shape: (5023, 3) or whatever you have

# === 加载原始 templates.pkl ===
print("📌 加载原始 templates.pkl...")
with open(original_templates_path, "rb") as f:
    templates = pickle.load(f, encoding="latin1")

# === 添加 biden 这个键 ===
print("📌 添加新 key: 'biden'")
templates["biden"] = vertices

# === 保存为新的 templates_biden.pkl ===
with open(output_templates_path, "wb") as f:
    pickle.dump(templates, f)

print("✅ 新模板已保存到:", output_templates_path)
