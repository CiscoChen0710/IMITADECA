import torch
import numpy as np
import os, datetime, glob
from timeit import default_timer as timer
from transformers import Wav2Vec2Processor
import librosa
import trimesh
from argparse import ArgumentParser
from pytorch_lightning import seed_everything
from imitator.utils.render_helper import render_helper
from imitator.utils.init_from_config import instantiate_from_config
from omegaconf import OmegaConf
from imitator.test.test_model_voca import get_latest_checkpoint

def load_obj_vertices(filepath):
    vertices = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) == 4:
                    x, y, z = map(float, parts[1:])
                    vertices.append([x, y, z])
    return np.array(vertices, dtype=np.float32)

class TestOnBidenAudio:
    def __init__(self, template_obj_path):
        self.selected_indices = np.load("D:/Purdue/ECE57000/reimplemention/Imitator/personalized_Ebiden/refined_selected_final_v3.npy")

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.rh = render_helper()

        # 加载顶点 & faces（只用 trimesh 加载一次为的是拿 faces）
        mesh = trimesh.load(template_obj_path, process=False)
        self.faces = mesh.faces  # ✅ 保存 faces

        # 再用你自己手动加载方法读取 vertices（保证顺序一致）
        self.vertices = load_obj_vertices(template_obj_path)
        print("✅ vertices shape:", self.vertices.shape)
        print("✅ faces shape:", self.faces.shape)
        print("🧾 faces[0:5]:", self.faces[:5])

        self.templates = {"biden": self.vertices}


    def read_audio(self, wav_path):
        speech_array, _ = librosa.load(wav_path, sr=16000)
        input_values = np.squeeze(self.processor(speech_array, sampling_rate=16000).input_values)
        return input_values

    def load_model(self, model_ckpt):
        logdir = os.path.dirname(os.path.dirname(model_ckpt))
        config_paths = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        configs = [OmegaConf.load(cfg) for cfg in config_paths]
        configs = OmegaConf.merge(*configs)
        model = instantiate_from_config(configs.model)
        # ✅ 如有需要加载 checkpoint，请取消下一行注释
        model.init_from_ckpt(model_ckpt)
        return model

    def run(self, model, audio_file, subj_name, out_dir):
        # Step 1: 加载音频并编码
        audio = self.read_audio(audio_file)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()  # [1, T]

        # Step 2: 加载 template
        template = torch.from_numpy(self.templates[subj_name]).float()  # [V, 3]
        template_flat = template.reshape(1, -1).contiguous()  # [1, V*3]

        # Step 3: 准备 one-hot（因为你只有一个人）
        one_hot = torch.tensor([[1.0]], dtype=torch.float32)

        # Step 4: 模型预测 displacement（[1, T, V*3]）
        prediction = model.nn_model.predict(audio_tensor, template_flat, one_hot)  # [1, T, V*3]
        prediction = prediction.squeeze(0)  # [T, V*3]

        # Step 5: reshape 并加上 template 得到最终 mesh
        prediction = prediction.view(prediction.shape[0], -1, 3)  # [T, V, 3]
        template_repeat = template.unsqueeze(0).repeat(prediction.shape[0], 1, 1)  # [T, V, 3]
        prediction = prediction + template_repeat  # [T, V, 3]

        print("🧪 prediction.min():", prediction.min().item())
        print("🧪 prediction.max():", prediction.max().item())
        print("🧪 prediction.mean():", prediction.mean().item())
        print("🧪 first frame center:", prediction[0].mean(0).tolist())
        print("🧪 last frame center:", prediction[-1].mean(0).tolist())


        # ✅ 使用选定索引（嘴巴+眉毛）计算运动幅度
        frame0 = prediction[0]
        frameN = prediction[-1]
        displacement = (frameN - frame0)[self.selected_indices]
        print("👉 Selected region (mouth + brows) movement:", displacement.abs().mean().item())

        full_motion = (frameN - frame0).abs().mean().item()
        print(f"🧪 Full face motion avg: {full_motion:.6f} vs. Mouth+Brow: {displacement.abs().mean().item():.6f}")

        # Step 6: detach 并保存视频
        prediction = prediction.detach()

        # Step 7: 输出 mesh + 音频视频
        seq_name = f"{subj_name}_{os.path.basename(audio_file).split('.')[0]}"
        self.rh.visualize_meshes(out_dir, seq_name, prediction, audio_file)

        # Debug 输出
        print(">>> prediction shape:", prediction.shape)
        print(">>> prediction min:", prediction.min().item())
        print(">>> prediction max:", prediction.max().item())
        print(">>> prediction example (first vertex of first frame):", prediction[0][0])

        self.rh.visualize_meshes(out_dir, seq_name, prediction, audio_file, faces=self.faces)
        import trimesh

        save_frames = [0, prediction.shape[0] // 2, -1]  # 保存第 0 帧、中间帧、最后一帧
        for idx in save_frames:
            mesh = trimesh.Trimesh(vertices=prediction[idx].cpu().numpy(), faces=self.faces)
            save_path = os.path.join(out_dir, f"{subj_name}_{idx:03d}.obj")
            mesh.export(save_path)
            print(f"✅ Saved frame {idx} to {save_path}")

        import trimesh
        save_frames = [0, prediction.shape[0] // 2, -1]
        for idx in save_frames:
            mesh = trimesh.Trimesh(vertices=prediction[idx].cpu().numpy(), faces=self.faces)
            save_path = os.path.join(out_dir, f"check_frame_{idx:03d}.obj")
            mesh.export(save_path)
            print(f"✅ Saved check frame to {save_path}")
        
        out_npy_path = os.path.join(out_dir, f"{subj_name}_{os.path.basename(audio_file).split('.')[0]}.npy")
        np.save(out_npy_path, prediction.cpu().numpy())
        print(f"✅ Saved prediction npy to: {out_npy_path}")



        return prediction



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to checkpoint .ckpt file")
    parser.add_argument("-a", "--audio", type=str, required=True, help="Path to input .wav file")
    parser.add_argument("-t", "--template", type=str, required=True, help="Path to template.obj")
    parser.add_argument("-o", "--out_dir", type=str, default="output", help="Directory to save results")
    return parser.parse_args()


if __name__ == "__main__":
    start = timer()
    args = parse_args()
    seed_everything(23)
    tester = TestOnBidenAudio(args.template)
    os.makedirs(args.out_dir, exist_ok=True)
    model = tester.load_model(args.model)
    model.eval()
    tester.run(model, args.audio, "biden", args.out_dir)
    end = timer()
    print("\n Done! Total time: %.2fs" % (end - start))
