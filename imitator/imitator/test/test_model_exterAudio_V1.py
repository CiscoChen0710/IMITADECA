import torch
import numpy as np
import os, datetime, glob
from timeit import default_timer as timer
from transformers import Wav2Vec2Processor
import librosa
import trimesh
from omegaconf import OmegaConf
from argparse import ArgumentParser
from pytorch_lightning import seed_everything
from imitator.test.test_model_voca import get_latest_checkpoint
from imitator.utils.render_helper import render_helper
from imitator.utils.init_from_config import instantiate_from_config

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

class test_on_audio():

    def __init__(self, template_path):
        if os.getenv("WAV2VEC_PATH"):
            wav2vec_path = os.getenv("WAV2VEC_PATH")
        else:
            wav2vec_path = "facebook/wav2vec2-base-960h"
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)
        self.rh = render_helper()

        # ✅ 加载 faces（只用 trimesh 拿 face，不要拿 vertices）
        print(f"Loading custom template from: {template_path}")
        mesh = trimesh.load(template_path, process=False)
        self.faces = mesh.faces

        # ✅ 用手动方式加载 vertices，确保顺序不变
        self.template = load_obj_vertices(template_path)

    def read_audio_from_file(self, wav_path):
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        input_values = np.squeeze(self.processor(speech_array, sampling_rate=16000).input_values)
        return input_values

    def load_model_from_checkpoint(self, model_ckpt):
        def find_index_from_list_with_partial_match(path_list, search_str):
            for i, sub_string in enumerate(path_list):
                if search_str in sub_string:
                    break
            return i

        paths = model_ckpt.split("/")
        idx = len(paths) - find_index_from_list_with_partial_match(paths[::-1], "log") + 1
        logdir = "/".join(paths[:idx])
        print("Logdir", logdir)
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        print("base_configs", base_configs)
        configs = [OmegaConf.load(cfg) for cfg in base_configs]
        configs = OmegaConf.merge(*configs)

        model = instantiate_from_config(configs.model)
        model.init_from_ckpt(model_ckpt)
        return model

    def run_on_wav_file(self, model, audio_file, condition_for_testing, out_dir):
        processed_audio = self.read_audio_from_file(audio_file)
        sampled_processed_audio = torch.from_numpy(processed_audio).view(1, -1)

        template = torch.from_numpy(self.template).view(1, -1)
        file_name = os.path.basename(audio_file)

        result_npy_dict = {}
        for condition in condition_for_testing:
            seq_name_w_condition = f"custom_{file_name.split('.wav')[0]}_condition_{condition}"
            print("seq name with condition", seq_name_w_condition)

            one_hot = torch.tensor([[1.0]])  # 只一个 identity

            prediction = model.nn_model.predict(sampled_processed_audio, template, one_hot)
            prediction = prediction.reshape(prediction.shape[1], -1, 3).detach()

            self.rh.visualize_meshes(out_dir, seq_name_w_condition, prediction, audio_file)
            result_npy_dict[seq_name_w_condition] = prediction.cpu().numpy()

        return result_npy_dict

    def run_test(self, model_ckpt, audio_path, out_dir, condition):
        model = self.load_model_from_checkpoint(model_ckpt)
        model.eval()
        os.makedirs(out_dir, exist_ok=True)
        return self.run_on_wav_file(model, audio_path, [condition], out_dir)

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-a', '--audio', type=str, required=True)
    parser.add_argument('-t', '--template', type=str, required=True)
    parser.add_argument('-c', '--condition', type=int, default=2)
    parser.add_argument('-o', '--out_dir', type=str, default="output")
    parser.add_argument('-s', '--seed', type=int, default=23)
    return parser

if __name__ == "__main__":
    start = timer()
    parser = get_parser()
    opt = parser.parse_args()
    seed_everything(opt.seed)

    tester = test_on_audio(opt.template)
    tester.run_test(opt.model, opt.audio, opt.out_dir, opt.condition)

    end = timer()
    print("\nDone! Total time: %.2fs" % (end - start))
