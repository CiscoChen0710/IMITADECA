import pickle
import numpy as np

# 加载 PKL 文件中的模板
with open('templates_biden.pkl', 'rb') as f:
    templates = pickle.load(f)

# 取出 'biden' 对应的模板顶点，假设是 ndarray，shape 为 (5023, 3)
vertices_from_pkl = templates['biden']
print(f'From PKL: {vertices_from_pkl.shape}')

def load_obj_vertices(filepath):
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) == 4:
                    x, y, z = map(float, parts[1:])
                    vertices.append([x, y, z])
    return np.array(vertices)

vertices_from_obj = load_obj_vertices('template_latest_v1.obj')
print(f'From OBJ: {vertices_from_obj.shape}')

if vertices_from_pkl.shape != vertices_from_obj.shape:
    print("❌ 顶点数量不一致！")
else:
    diff = np.abs(vertices_from_pkl - vertices_from_obj)
    max_diff = np.max(diff)
    if max_diff < 1e-6:
        print("✅ 两个模板完全一致！")
    else:
        print(f"⚠️ 模板不一致，最大差异: {max_diff}")
