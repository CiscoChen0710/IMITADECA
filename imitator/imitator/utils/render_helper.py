import os
import sys
from tqdm import tqdm
import cv2
import numpy as np
from imitator.utils.util_pyrenderer import Facerender
from FLAMEModel.FLAME import FLAME
import trimesh
import torch



class render_helper():
    def __init__(self, config={}):
        if not config:
            config["flame_model_path"] = os.path.join("FLAMEModel/model/generic_model.pkl")
            config["batch_size"] = 1
            config["shape_params"] = 0
            config["expression_params"] = 100
            config["pose_params"] = 0
            config["number_worker"] = 8
            config["use_3D_translation"] = False

        self.face_model = FLAME(config)
        self.image_size = (512, 512)
        self.face_render = Facerender()

    def visualize_meshes(self, out_dir, out_seq_name, pred_vertices, audio_file=None, faces=None):
        import trimesh

        out_pred_rendered_images = []
        export_obj_dir = os.path.join(out_dir, "exported_objs")
        os.makedirs(export_obj_dir, exist_ok=True)

        for i in tqdm(range(pred_vertices.shape[0]), desc="Rendering expressions"):
            self.face_render.reset_scene()
            pred_frame = self.render_images(pred_vertices[i], faces=faces)
            out_pred_rendered_images.append(pred_frame)

            # ✅ 只导出第 0 帧和最后一帧
            if i in [0, pred_vertices.shape[0] - 1]:
                verts_np = pred_vertices[i].cpu().numpy() if isinstance(pred_vertices[i], torch.Tensor) else pred_vertices[i]
                mesh = trimesh.Trimesh(vertices=verts_np, faces=faces if faces is not None else self.face_model.faces, process=False)
                output_path = os.path.join(export_obj_dir, f"{out_seq_name}_frame_{i:03d}.obj")
                mesh.export(output_path)
                print(f"Exported {output_path}")

        out_vid_file = os.path.join(out_dir, out_seq_name + ".mp4")
        video_file = self.compose_write_video(out_vid_file, out_pred_rendered_images, self.image_size)

        if audio_file is not None:
            out_vid_w_audio_file = os.path.join(out_dir, out_seq_name + "_wAudio.mp4")
            self.add_audio_to_video(audio_file, video_file, out_vid_w_audio_file)
            out_vid_file = out_vid_w_audio_file

        return out_vid_file, out_pred_rendered_images



    def render_images(self, vertices, faces=None):
        verts = vertices.cpu().numpy()
            #添加缩放因子（比如放大 2 倍）
        scale_factor = 1.0
        verts *= scale_factor
        self.face_render.add_face(verts, faces if faces is not None else self.face_model.faces)
        colour = self.face_render.render()
        return colour

    def compose_write_video(self, out_vid_file, gt_frames, frame_size=(512, 512), fps=30):
        print('Writing video', out_vid_file)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter(out_vid_file, fourcc, fps, frame_size)
        for frame in tqdm(gt_frames, desc="Writing video"):
            writer.write(frame)
        writer.release()
        return out_vid_file

    def add_audio_to_video(self, audio_file, video_file, out_vid_w_audio_file):
        print("Audio file", audio_file)
        ffmpeg_command = f"ffmpeg -y -i {video_file} -i {audio_file} -map 0:v -map 1:a -c:v copy -shortest {out_vid_w_audio_file}"
        os.system(ffmpeg_command)
        print("Added audio to:", out_vid_w_audio_file)

        if sys.platform.startswith('win'):
            rm_command = f'del "{video_file}"'
        else:
            rm_command = f'rm {video_file}'
        os.system(rm_command)
