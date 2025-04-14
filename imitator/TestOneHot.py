import torch
from imitator.models.nn_model import imitator

# 设置路径（你自己的风格化模型）
ckpt_path = "logs/biden_style_adapt_stage2/version_4/checkpoints/epoch=29-step=240.ckpt"

# 加载 state_dict
ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

# 解析 nn_model 配置（你要根据训练用的 project.yaml 里的参数填写）
nn_model_params = {
    "dataset": "biden",
    "vertice_dim": 15069,
    "feature_dim": 64,
    "period": 30,
    "gradient_accumulation_steps": 1,
    "num_identity_classes": 1,
    "train_teacher_forcing": False,
    "wav2vec_model": "pretrained/wav2vec2-base-960h",
    "wav2vec_static_features": False,
    "num_dec_layers": 5,
    "fixed_channel": True,
    "style_concat": False,
    "train_subjects": "biden"
}

# 初始化模型并加载参数
model = imitator(**nn_model_params)
model.load_state_dict(state_dict, strict=False)

# 打印 obj_vector 相关信息
print("🧠 obj_vector.weight shape:", model.obj_vector.weight.shape)

# 模拟测试时 one-hot
num_classes = model.obj_vector.in_features
bs = 1
test_one_hot = torch.zeros(bs, num_classes)
test_one_hot[:, 0] = 1

print("🧪 one-hot used during test:", test_one_hot.numpy())
