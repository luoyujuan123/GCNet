# stable_video_generator.py
import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import numpy as np

class StableVideoGenerator:
    def __init__(self, model_id="stabilityai/stable-video-diffusion-img2vid"):
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.pipe.enable_model_cpu_offload()
    
    def generate_from_image(self, image_path, num_frames=25, fps=6):
        """从图片生成视频（10秒约60帧）"""
        # 加载初始图片
        init_image = Image.open(image_path)
        
        # 生成视频帧
        frames = self.pipe(
            init_image,
            decode_chunk_size=8,
            motion_bucket_id=127,
            noise_aug_strength=0.1,
            num_frames=num_frames,
        ).frames[0]
        
        # 保存为视频
        self.save_frames_as_video(frames, fps=fps)
        return frames
    
    def generate_from_text(self, prompt, num_frames=25):
        """文本生成视频（需要先文生图）"""
        # 先用文生图模型生成第一帧
        from diffusers import StableDiffusionPipeline
        
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        init_image = sd_pipe(prompt).images[0]
        
        # 再用SVD生成视频
        return self.generate_from_image(init_image, num_frames)