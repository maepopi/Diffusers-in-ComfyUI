from diffusers import AutoencoderKL, StableDiffusionXLPipeline, StableDiffusionPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline, ControlNetModel
import folder_paths
import torch
import numpy as np
import os
from torchvision.transforms import ToTensor
from diffusers.utils import load_image, make_image_grid
import cv2
from PIL import Image

'''To do
- Investigate the model caching
- Deduce the model architecture, SDXL or SD (see B-Lora Comfy code)
'''



class GenerateStableDiffusionPipeline:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "is_sdxl": ("BOOLEAN", {"default": True}),
                    "low_vram": ("BOOLEAN", {"default": True}),
                    "model": (folder_paths.get_filename_list("checkpoints"),),
                },
                "optional":
                {
                    "vae": ("STRING", {"multiline": False}),
                    "controlnet_model" : ("STRING", {"multiline": False}),
                }
                }

    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "create_pipeline"
    CATEGORY = "Diffusers-in-Comfy"


    def create_pipeline(self, is_sdxl, low_vram, model, vae, controlnet_model):
        args = {
            "pretrained_model_link_or_path" : folder_paths.get_full_path("checkpoints", model),
            "torch_dtype" : torch.float16,
        }

        if vae != '':
            args['vae'] =  AutoencoderKL.from_pretrained(vae, torch_dtype=torch.float16, use_safetensors=True)

        if controlnet_model != '':
            args['controlnet'] = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16, use_safetensors=True)


        if is_sdxl:
            if 'controlnet' in args:
                pipeline = StableDiffusionXLControlNetPipeline.from_single_file(**args)
                
            else:
                pipeline = StableDiffusionXLPipeline.from_single_file(**args)

        else:
            if 'controlnet' in args:
                pipeline = StableDiffusionControlNetPipeline.from_single_file(**args)
                                    
            else:
                pipeline = StableDiffusionPipeline.from_single_file(**args)
                

        if low_vram:
            pipeline.enable_xformers_memory_efficient_attention()
            pipeline.enable_model_cpu_offload()

        return (pipeline,)
    


class MakeCanny:

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "image_path" : ("STRING", {"multiline": True}),
                    "high_threshold": ("INT", {"default": 200, "min":0, "max":255}),
                    "low_threshold": ("INT", {"default": 100, "min":0, "max":255}),

                }}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("Canny", "Canny Preview")
    FUNCTION = "create_canny"
    CATEGORY = "Diffusers-in-Comfy"

    def convert_images_to_tensors(self, images):
        return torch.stack([np.transpose(ToTensor()(image), (1, 2, 0)) for image in images])

    def create_canny(self, image_path, low_threshold, high_threshold):
        original_image = load_image(image_path)
        image = np.array(original_image)
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny = Image.fromarray(image)
        return (canny, self.convert_images_to_tensors([canny]),)
        


class ImageInference:
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
                    "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),      
                },

                "optional": {
                    "controlnet_image" : ("IMAGE",),
                    "controlnet_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step":0.1, "round": 0.01}),
                    
                } 
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Result",)
    FUNCTION = "infer_image"
    CATEGORY = "Diffusers-in-Comfy"

    def convert_images_to_tensors(self, images):
        return torch.stack([np.transpose(ToTensor()(image), (1, 2, 0)) for image in images])


    def infer_image(self, pipeline, seed, steps, cfg, positive, negative, width, height, controlnet_image=None, controlnet_scale=None):
        generator = torch.Generator(device='cuda').manual_seed(seed)

        print(f'at this stage, controlnet image is {controlnet_image} of type {type(controlnet_image)}')

        args = {
            "prompt": positive,
            "generator": generator,
            "num_inference_steps" : steps,
            "guidance_scale" : cfg,
            "width" : width,
            "height" : height,
        }

        if negative != '':
            args['negative'] = negative
        
        if controlnet_image:
            args['image'] = controlnet_image
            args['controlnet_conditioning_scale'] = controlnet_scale
        
        images = pipeline(**args).images

        return (self.convert_images_to_tensors(images),)



class LoRALoader:
    def __init__(self) -> None:
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
                "pipeline" : ("PIPELINE",),
                "lora_name" : (folder_paths.get_filename_list("loras"),),
                "lora_scale" : ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.1, "round": 0.01}),
            
        }}
    
    RETURN_TYPES = ("PIPELINE",) 
    FUNCTION = "load_lora"
    CATEGORY = "Diffusers-in-Comfy"

    def load_lora(self, pipeline, lora_name, lora_scale):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_name = os.path.splitext(lora_name)[0]
        pipeline.load_lora_weights(lora_path, adapter_name=lora_name)
        pipeline.set_adapters([lora_name], adapter_weights=[lora_scale])


        return (pipeline,)


class BLoRALoader:
    BLOCKS = {
    'content': ['unet.up_blocks.0.attentions.0'],
    'style': ['unet.up_blocks.0.attentions.1'],
    }

    def __init__(self) -> None:
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
                "pipeline" : ("PIPELINE",),
                "style_lora_name" : ("STRING", {"multiline": False}),
                "style_lora_scale" : ("FLOAT", {"default": 1.1, "min": 0.0, "max": 1.1, "step":0.1, "round": 0.01}),
                "content_lora_name" : ("STRING", {"multiline": False}),
                "content_lora_scale" : ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.1, "round": 0.01}),
              
        }}
    
    RETURN_TYPES = ("PIPELINE",) 
    FUNCTION = "load_b_lora_to_unet"
    CATEGORY = "Diffusers-in-Comfy"

    def is_belong_to_blocks(self, key, blocks):
        try:
            for g in blocks:
                if g in key:
                    return True
            return False
        except Exception as e:
            raise type(e)(f'failed to is_belong_to_block, due to: {e}')

    def filter_lora(self, state_dict, blocks_):
        try:
            return {k: v for k, v in state_dict.items() if self.is_belong_to_blocks(k, blocks_)}
        except Exception as e:
            raise type(e)(f'failed to filter_lora, due to: {e}')

    def scale_lora(self, state_dict, alpha):
        try:
            return {k: v * alpha for k, v in state_dict.items()}
        except Exception as e:
            raise type(e)(f'failed to scale_lora, due to: {e}')


    def load_b_lora_to_unet(self, 
                            pipeline, 
                            style_lora_name, 
                            content_lora_name,
                            style_lora_scale=1.,
                            content_lora_scale=1.):
        
        if style_lora_name != '' :
            style_lora_path = folder_paths.get_full_path("loras", style_lora_name)
            style_lora_state_dict, _ = pipeline.lora_state_dict(style_lora_path)
            style_lora = self.filter_lora(style_lora_state_dict, self.BLOCKS['style'])
            style_lora = self.scale_lora(style_lora, style_lora_scale)
        
        else:
            style_lora = {}
        

        if content_lora_name != '':
            content_lora_path = folder_paths.get_full_path("loras", content_lora_name)
            content_lora_state_dict, _ = pipeline.lora_state_dict(content_lora_path)
            content_lora = self.filter_lora(content_lora_state_dict, self.BLOCKS['content'])
            content_lora = self.scale_lora(content_lora, content_lora_scale)

        else:
            content_lora = {}

        merged_lora = {**style_lora, **content_lora}

    
        pipeline.load_lora_into_unet(merged_lora, None, pipeline.unet)


        return (pipeline,)



    

NODE_CLASS_MAPPINGS = {
    "CreatePipeline": GenerateStableDiffusionPipeline,
    "GenerateImage": ImageInference,
    "LoRALoader" : LoRALoader,
    "BLoRALoader" : BLoRALoader,
    "MakeCanny": MakeCanny,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CreatePipeline" : "GenerateStableDiffusionPipeline",
    "GenerateImage" : "ImageInference",
    "LoRALoader" : "LoRALoader",
    "BLoRALoader" : "BLoRALoader",
    "MakeCanny": "MakeCanny",
   
}