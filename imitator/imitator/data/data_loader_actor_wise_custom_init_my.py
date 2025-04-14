import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
import pytorch_lightning as pl
import glob

# 读取 .obj 文件中的顶点数据
def load_obj_vertices(filepath):
    vertices = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) == 4:
                    x, y, z = map(float, parts[1:])
                    vertices.append([x, y, z])
    return np.array(vertices)

class Dataset(data.Dataset):
    def __init__(self, data, subjects_dict, data_type="train", number_identity_cls=8, custom_init_id="biden", template_path=None):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(number_identity_cls)
        self.custom_init_id = custom_init_id
        # ✅ 确保模板路径为绝对路径
        self.template_path = os.path.abspath(template_path)
        # ✅ 只加载一次模板
        self.template = load_obj_vertices(self.template_path).reshape(-1)
        self.default_index = self.subjects_dict["train_subjects_all"].index(custom_init_id)

    def __getitem__(self, index):
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        one_hot = self.one_hot_labels[self.default_index]

        if vertice.ndim == 3 and vertice.shape[2] == 3:
            vertice = vertice.reshape(vertice.shape[0], -1)

        assert vertice.shape[-1] == self.template.shape[0], f"vertice dim: {vertice.shape[-1]}, template dim: {self.template.shape[0]}"
        # 只检测前几个 batch（防止输出过多）
        if index < 5:
            # 你的 mouth+eyebrow 区域 index
            selected_indices = np.load("D:/Purdue/ECE57000/reimplemention/Imitator/personalized_Ebiden/refined_selected_final_v3.npy")
            disp = vertice.reshape(vertice.shape[0], -1, 3)
            mouth_motion = disp[:, selected_indices]
            mouth_movement = np.abs(mouth_motion).mean()
            print(f"🟢 [{file_name}] mouth + brow movement avg: {mouth_movement:.6f}")


        return (
            torch.FloatTensor(audio),
            torch.FloatTensor(vertice),
            torch.FloatTensor(self.template),
            torch.FloatTensor(one_hot),
            file_name
        )

    def __len__(self):
        return self.len


def read_data(
        dataset,
        dataset_root,
        wav_path,
        vertices_path,
        template_file,
        train_subjects,
        val_subjects,
        test_subjects,
        **kwargs
):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    base_path = os.getenv("VOCASET_PATH") or os.getenv("HOME") or "."

    audio_path = os.path.join(base_path, dataset_root, wav_path)
    vertices_path = os.path.join(base_path, dataset_root, vertices_path)
    template_path = os.path.join(base_path, dataset_root, template_file)

    templates = {"biden": load_obj_vertices(template_path)}

    subjects_dict = {
        "train": train_subjects.split(" "),
        "val": val_subjects.split(" "),
        "test": test_subjects.split(" ")
    }

    all_subjects = set(subjects_dict["train"] + subjects_dict["val"] + subjects_dict["test"])
    for subj in all_subjects:
        audio_files = glob.glob(os.path.join(audio_path, subj + "*.wav"))
        for wav_file in tqdm(audio_files):
            fname = os.path.basename(wav_file)
            speech_array, _ = librosa.load(wav_file, sr=16000)
            input_values = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
            npy_name = fname.replace(".wav", ".npy")
            npy_path = os.path.join(vertices_path, npy_name)
            if not os.path.exists(npy_path):
                continue
            data[npy_name]["audio"] = input_values
            data[npy_name]["name"] = fname
            data[npy_name]["template"] = templates["biden"].reshape(-1)
            data[npy_name]["vertice"] = np.load(npy_path, allow_pickle=True)

    subjects_dict["train_subjects_all"] = kwargs.get("train_subjects_all", train_subjects).split(" ")

    for k, v in data.items():
        subject_id = "biden"
        if subject_id in subjects_dict["train"]:
            train_data.append(v)
        if subject_id in subjects_dict["val"]:
            valid_data.append(v)
        if subject_id in subjects_dict["test"]:
            test_data.append(v)

    print("Datset distribution")
    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        train_data, valid_data, test_data, subjects_dict = read_data(**kwargs)
        num_iden_cls = kwargs["num_iden_cls"]
        init_id = kwargs.get("default_init_subject", "biden")
        template_path = os.path.join(kwargs["dataset_root"], kwargs["template_file"])
        self.train_data = Dataset(train_data, subjects_dict, "train", num_iden_cls, init_id, template_path)
        self.valid_data = Dataset(valid_data, subjects_dict, "val", num_iden_cls, init_id, template_path)
        self.test_data = Dataset(test_data, subjects_dict, "test", num_iden_cls, init_id, template_path)

        self.data_cfg = kwargs

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return data.DataLoader(dataset=self.train_data, batch_size=self.data_cfg.get('batch_size', 8), shuffle=True, num_workers=self.data_cfg.get('num_workers', 0))

    def val_dataloader(self):
        return data.DataLoader(dataset=self.valid_data, batch_size=self.data_cfg.get('batch_size', 8), shuffle=False, num_workers=self.data_cfg.get('num_workers', 0))

    def test_dataloader(self):
        return data.DataLoader(dataset=self.test_data, batch_size=self.data_cfg.get('batch_size', 8), shuffle=False, num_workers=self.data_cfg.get('num_workers', 0))

    def test_unseen_dataloader(self):
        return None
