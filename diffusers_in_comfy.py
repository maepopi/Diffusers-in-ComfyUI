from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import folder_paths
import torch
import numpy as np
from torchvision.transforms import ToTensor


class CreatePipeline:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "is_sdxl": ("BOOLEAN", {"default": True}),
                    "model": (folder_paths.get_filename_list("checkpoints"),),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps":  ("INT", {"default": 50, "min": 1, "max": 10000}),
                    
                }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "Diffusers-in-Comfy"

    def convert_images_to_tensors(self, images):
        return torch.stack([np.transpose(ToTensor()(image), (1, 2, 0)) for image in images])


    def generate_image(self, pipeline, seed, steps):
        generator = torch.Generator(device='cuda').manual_seed(seed)
        prompt = "a cat sleeping on a cushion"
        images = pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps = steps
        ).images

        return (self.convert_images_to_tensors(images),)
        
    
    def main(self, is_sdxl, model, seed, steps):
        
        model_path = folder_paths.get_full_path("checkpoints", model)

        if is_sdxl:
            pipeline = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16).to("cuda")
        
        else:
            pipeline =  StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16).to("cuda")
        

        return self.generate_image(pipeline, seed, steps)


NODE_CLASS_MAPPINGS = {
    "CreatePipeline": CreatePipeline,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CreatePipeline" : "CreatePipeline",
   
}