
import os
import numpy as np
import trimesh

# === 配置路径 ===
template_path = "template_fixed_5023_faces.obj"
displacement_dir = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\disps_full_nose_eyebrow"
output_mesh_dir = r"D:\Purdue\ECE57000\reimplemention\Imitator\personalized_Ebiden\preview_meshes_fullface"
os.makedirs(output_mesh_dir, exist_ok=True)

# === 加载模板顶点和面 ===
def load_obj_vertices_faces(filepath):
    vertices = []
    faces = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                parts = line.strip().split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                if len(face) == 3:
                    faces.append(face)
    return np.array(vertices), np.array(faces)

template_vertices, template_faces = load_obj_vertices_faces(template_path)

# === 导出前几帧预览 mesh ===
disps = sorted([f for f in os.listdir(displacement_dir) if f.endswith(".npy")])
preview_files = disps[:5]

for fname in preview_files:
    disp = np.load(os.path.join(displacement_dir, fname))
    recovered = template_vertices + disp
    mesh = trimesh.Trimesh(vertices=recovered, faces=template_faces, process=False)
    out_path = os.path.join(output_mesh_dir, fname.replace(".npy", ".ply"))
    mesh.export(out_path)
    print(f"✅ 导出 mesh: {out_path}")
