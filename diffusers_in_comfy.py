from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import folder_paths
import torch
import numpy as np
from torchvision.transforms import ToTensor

'''To do
- Investigate the model caching
- Add options for low VRAM
'''




class CreatePipeline:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "is_sdxl": ("BOOLEAN", {"default": True}),
                    "low_vram": ("BOOLEAN", {"default": True}),
                    "model": (folder_paths.get_filename_list("checkpoints"),),
                }}

    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "create_pipeline"
    CATEGORY = "Diffusers-in-Comfy"


    
    def create_pipeline(self, is_sdxl, low_vram, model):
        
        model_path = folder_paths.get_full_path("checkpoints", model)

        if is_sdxl:
            pipeline = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16).to("cuda")
        
        else:
            pipeline = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16).to("cuda")
        
        if low_vram:
            pipeline.enable_xformers_memory_efficient_attention()
            pipeline.enable_model_cpu_offload()
        
        return (pipeline,)

        


class GenerateImage:
    def __init__(self) -> None:
        pass


    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "pipeline": ("PIPELINE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "positive": ("STRING", {"multiline": True}),
                    "negative": ("STRING", {"multiline": True}),
                    "steps":  ("INT", {"default": 50, "min": 1, "max": 10000}),
                    "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                    "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),       
                }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "Diffusers-in-Comfy"

    def convert_images_to_tensors(self, images):
        return torch.stack([np.transpose(ToTensor()(image), (1, 2, 0)) for image in images])


    def generate_image(self, pipeline, seed, steps, cfg, positive, negative, width, height,):
        generator = torch.Generator(device='cuda').manual_seed(seed)
        images = pipeline(
            prompt=positive,
            generator=generator,
            negative_prompt = negative,
            num_inference_steps = steps,
            guidance_scale = cfg,
            width=width,
            height=height,
        ).images

        return (self.convert_images_to_tensors(images),)
        



NODE_CLASS_MAPPINGS = {
    "CreatePipeline": CreatePipeline,
    "GenerateImage": GenerateImage,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CreatePipeline" : "CreatePipeline",
    "GenerateImage" : "GenerateImage",
   
}