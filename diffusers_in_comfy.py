from .utils import PipelineFactory, Text2ImgPipelineCreator, Img2ImgPipelineCreator, InpaintPipelineCreator
from .utils import Text2ImageInference, InpaintInference, Img2ImgInference
from .utils import convert_images_to_tensors, filter_lora, scale_lora, invert_mask
from diffusers.utils import load_image
import folder_paths
import torch
import numpy as np
import os
import cv2

from PIL import Image

'''To do
- Investigate the model caching
- Deduce the model architecture, SDXL or SD (see B-Lora Comfy code)
- change pipeline instanciation
'''


# Registering the needed pipelines
PipelineFactory.register_creator('text2img', Text2ImgPipelineCreator)
PipelineFactory.register_creator('img2img', Img2ImgPipelineCreator)
PipelineFactory.register_creator('inpainting', InpaintPipelineCreator)




class Text2ImgStableDiffusionPipeline:
    '''
        This node generates the Stable Diffusion Pipeline for Text2Img.

        Required inputs : 
            - is_sdxl : whether the loaded model has an SDXL base
            - low_vram : whether to activate low VRAM options 
            - model :  the model that has to be instanciated

        Optional outputs:
            - vae : which VAE to use - copy the name of the VAE from Hugging Face
            - controlnet_model : which ControlNet model to use - copy the name of the ControlNet from Hugging Face

        Ouputs:
            Pipeline
    
    '''
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
    FUNCTION = "create_text2img_pipeline"
    CATEGORY = "Diffusers-in-Comfy/Pipelines"


    def create_text2img_pipeline(self, is_sdxl, low_vram, model, vae, controlnet_model):
        
        factory = PipelineFactory()
        pipeline = factory.get_pipeline(low_vram, 'text2img', model, vae, controlnet_model, is_sdxl, torch_dtype=torch.float16)
        
        return (pipeline,)

class Img2ImgStableDiffusionPipeline:
    '''
        This node generates the Stable Diffusion Pipeline for Img2Img.

        Required inputs : 
            - is_sdxl : whether the loaded model has an SDXL base
            - low_vram : whether to activate low VRAM options 
            - model :  the model that has to be instanciated


        Optional outputs:
            - vae : which VAE to use - copy the name of the VAE from Hugging Face
            - controlnet_model : which ControlNet model to use - copy the name of the ControlNet from Hugging Face

        Ouputs:
            Pipeline
    
    '''
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
    FUNCTION = "create_img2img_pipeline"
    CATEGORY = "Diffusers-in-Comfy/Pipelines"


    def create_img2img_pipeline(self, is_sdxl, low_vram, model, vae, controlnet_model):
        factory = PipelineFactory()
        pipeline = factory.get_pipeline(low_vram, 'img2img', model, vae, controlnet_model, is_sdxl, torch_dtype=torch.float16)

        return (pipeline,)
    
class InpaintingStableDiffusionPipeline:
    '''
        This node generates the Stable Diffusion Pipeline for inpainting.

        Required inputs : 
            - is_sdxl : whether the loaded model has an SDXL base
            - low_vram : whether to activate low VRAM options 
            - model :  the model that has to be instanciated


        Optional outputs:
            - vae : which VAE to use - copy the name of the VAE from Hugging Face
            - controlnet_model : which ControlNet model to use - copy the name of the ControlNet from Hugging Face

        Ouputs:
            Pipeline
    
    '''
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
    FUNCTION = "create_inpaint_pipeline"
    CATEGORY = "Diffusers-in-Comfy/Pipelines"


    def create_inpaint_pipeline(self, is_sdxl, low_vram, model, vae, controlnet_model):
        factory = PipelineFactory()
        pipeline = factory.get_pipeline(low_vram, 'inpainting', model, vae, controlnet_model, is_sdxl, torch_dtype=torch.float16)

        return (pipeline,)


        
    

class MakeCanny:
    '''
        This node allows you to transform your image into a canny mask.

        Required inputs:
            - image_path : the string path to your image on your computer, or an URL
            - high_threshold : high threshold for the edges to be calculated
            - low_threshold : low_threshold for the edges to be calculated
        
        Output:
            - an image "Canny" that can then be processed further down the Comfy Workflow
            - a tensor 'Canny Preview" that is an image converted to tensor to be displayed by Comfy's preview image node
    '''

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
    CATEGORY = "Diffusers-in-Comfy/Utils"


    def create_canny(self, image_path, low_threshold, high_threshold):
        original_image = load_image(image_path)
        image = np.array(original_image)
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny = Image.fromarray(image)
        return (canny, convert_images_to_tensors([canny]),)
        

class GenerateInpaintImage:
    """
        This class proceeds to the inference of the image based on the pipeline and its components.

        Required inputs:
            - pipeline : the Stable Diffusion Pipeline
            - seed : seed for the generation of the image
            - positive : the positive prompt
            - negative : the negative prompt
            - steps : the number of inference steps
            - width : width of the generated image
            - height : height of the generated image
            - cfg : guidance scale
        
        Optional inputs:
            - controlnet_image : the image coming out of the Canny Node
            - controlnet_scale : weight of the canny to control the inferred image
            - input_image : the image that has to be inpainted
            - mask_image : the mask of the area where the image has to be inpainted

        Output:
            - an image
    """
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
                    "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "input_image" :  ("STRING", {"multiline": True}),
                    "mask_image" :  ("STRING", {"multiline": True}),
                    "mask_invert" : ("BOOLEAN",{"default":True}),      
                },

                "optional": {
                    "controlnet_image" : ("IMAGE",),
                    "controlnet_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step":0.1, "round": 0.01}),
                    
                } 
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Result",)
    FUNCTION = "generate_image"
    CATEGORY = "Diffusers-in-Comfy/Inference"

    def generate_image(self, 
                    pipeline, 
                    seed, 
                    steps, 
                    cfg, 
                    positive, 
                    negative, 
                    width, 
                    height,
                    input_image,
                    mask_image,
                    mask_invert,
                    controlnet_image=None, 
                    controlnet_scale=None, 
                    ):
     
        inpaint_inferer = InpaintInference()
        print(f' at this stage input image is {input_image}')
        images = inpaint_inferer.infer_image(pipeline, 
                                                  seed, 
                                                  steps, 
                                                  cfg, 
                                                  positive, 
                                                  negative, 
                                                  width, 
                                                  height,
                                                  input_image,
                                                  mask_image,
                                                  mask_invert,
                                                  controlnet_image=controlnet_image, 
                                                  controlnet_scale=controlnet_scale,  
                                                  )
        

        return (images,)
    
class GenerateImg2Image:
    """
        This class proceeds to the inference of the image based on the pipeline and its components.

        Required inputs:
            - pipeline : the Stable Diffusion Pipeline
            - seed : seed for the generation of the image
            - positive : the positive prompt
            - negative : the negative prompt
            - steps : the number of inference steps
            - width : width of the generated image
            - height : height of the generated image
            - cfg : guidance scale
            - input_image : the image that is going to be used as prompt
        
        Optional inputs:
            - controlnet_image : the image coming out of the Canny Node
            - controlnet_scale : weight of the canny to control the inferred image


        Output:
            - an image
    """
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
                    "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "input_image" :  ("STRING", {"multiline": True}),
                },

                "optional": {
                    "controlnet_image" : ("IMAGE",),
                    "controlnet_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step":0.1, "round": 0.01}),
                    
                } 
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Result",)
    FUNCTION = "generate_image"
    CATEGORY = "Diffusers-in-Comfy/Inference"

    def generate_image(self, 
                    pipeline, 
                    seed, 
                    steps, 
                    cfg, 
                    positive, 
                    negative, 
                    width, 
                    height,
                    input_image,
                    controlnet_image=None, 
                    controlnet_scale=None, 
                    ):
     
        img2img_inferer = Img2ImgInference()
        images = img2img_inferer.infer_image(pipeline, 
                                                  seed, 
                                                  steps, 
                                                  cfg, 
                                                  positive, 
                                                  negative, 
                                                  width, 
                                                  height,
                                                  input_image,
                                                  controlnet_image=controlnet_image, 
                                                  controlnet_scale=controlnet_scale, 
                                                  )
        

        return (images,)

class GenerateText2Image:
    """
        This class proceeds to the inference of the image based on the pipeline and its components.

        Required inputs:
            - pipeline : the Stable Diffusion Pipeline
            - seed : seed for the generation of the image
            - positive : the positive prompt
            - negative : the negative prompt
            - steps : the number of inference steps
            - width : width of the generated image
            - height : height of the generated image
            - cfg : guidance scale
        
        Optional inputs:
            - controlnet_image : the image coming out of the Canny Node
            - controlnet_scale : weight of the canny to control the inferred image

        Output:
            - an image
    """
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
                    "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),      
                },

                "optional": {
                    "controlnet_image" : ("IMAGE",),
                    "controlnet_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step":0.1, "round": 0.01}),
                } 
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Result",)
    FUNCTION = "generate_image"
    CATEGORY = "Diffusers-in-Comfy/Inference"

    def generate_image(self, 
                    pipeline, 
                    seed, 
                    steps, 
                    cfg, 
                    positive, 
                    negative, 
                    width, 
                    height, 
                    controlnet_image=None, 
                    controlnet_scale=None):
     
        txt2img_inferer = Text2ImageInference()
        images = txt2img_inferer.infer_image(pipeline, 
                                                  seed, 
                                                  steps, 
                                                  cfg, 
                                                  positive, 
                                                  negative, 
                                                  width, 
                                                  height, 
                                                  controlnet_image=controlnet_image, 
                                                  controlnet_scale=controlnet_scale)
        

        return (images,)


class LoRALoader:
    """
        This node allows to load a LoRA to the pipeline.

        Required inputs:
            - pipeline : the Stable diffusion pipeline to connect the LoRA to
            - lora_name : name of your lora in the Lora folder. Only write the namen with the extension, not the total path
            - lora_scale : weight of the lora on the generated output
        
        Output:
            - pipeline with the LoRA loaded to the Unet

    """
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
    CATEGORY = "Diffusers-in-Comfy/Components"

    def load_lora(self, pipeline, lora_name, lora_scale):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_name = os.path.splitext(lora_name)[0] # to remove the extension from the name
        pipeline.load_lora_weights(lora_path, adapter_name=lora_name)
        pipeline.set_adapters([lora_name], adapter_weights=[lora_scale])



        return (pipeline,)


class BLoRALoader:
    """
        This node allows to load a B-Lora to the Unet. 

        Required:
            - pipeline : the Stable Diffusion pipeline to which the B-Lora must be applied
            - style_lora_name : name of the lora you want to apply as style lora (only write the name + extension inside your lora folder, not total path)
            - style_lora_scale : weight of the style lora on the generated output
            - content_lora_name : name of the lora you want to apply as content lora (only write the name + extension inside your lora folder, not total path)
            - content_lora_scale : weight of the content lora on the generated output
        
        Output:
            - pipeline with the B-LoRA loaded into its Unet.
    """
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
    CATEGORY = "Diffusers-in-Comfy/Components"

    


    def load_b_lora_to_unet(self, 
                            pipeline, 
                            style_lora_name, 
                            content_lora_name,
                            style_lora_scale=1.,
                            content_lora_scale=1.):
        
        if style_lora_name != '' :
            style_lora_path = folder_paths.get_full_path("loras", style_lora_name)
            style_lora_state_dict, _ = pipeline.lora_state_dict(style_lora_path)
            style_lora = filter_lora(style_lora_state_dict, self.BLOCKS['style'])
            style_lora = scale_lora(style_lora, style_lora_scale)
        
        else:
            style_lora = {}
        

        if content_lora_name != '':
            content_lora_path = folder_paths.get_full_path("loras", content_lora_name)
            content_lora_state_dict, _ = pipeline.lora_state_dict(content_lora_path)
            content_lora = filter_lora(content_lora_state_dict, self.BLOCKS['content'])
            content_lora = scale_lora(content_lora, content_lora_scale)

        else:
            content_lora = {}

        merged_lora = {**style_lora, **content_lora}

    
        pipeline.load_lora_into_unet(merged_lora, None, pipeline.unet)


        return (pipeline,)



    

NODE_CLASS_MAPPINGS = {
    "Text2ImgStableDiffusionPipeline": Text2ImgStableDiffusionPipeline,
    "Img2ImgStableDiffusionPipeline" : Img2ImgStableDiffusionPipeline,
    "InpaintingStableDiffusionPipeline": InpaintingStableDiffusionPipeline,
    "GenerateTxt2Image" : GenerateText2Image,
    "GenerateInpaintImage": GenerateInpaintImage,
    "GenerateImg2Image" : GenerateImg2Image,
    "LoRALoader" : LoRALoader,
    "BLoRALoader" : BLoRALoader,
    "MakeCanny": MakeCanny,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Text2ImgStableDiffusionPipeline" : "Text2ImgStableDiffusionPipeline",
    "Img2ImgStableDiffusionPipeline" : "Img2ImgStableDiffusionPipeline",
    "InpaintingStableDiffusionPipeline": "InpaintingStableDiffusionPipeline",
    "GenerateTxt2Image" : "GenerateTxt2Image",
    "GenerateInpaintImage": "GenerateInpaintImage",
    "GenerateImg2Image" : "GenerateImg2Image",
    "LoRALoader" : "LoRALoader",
    "BLoRALoader" : "BLoRALoader",
    "MakeCanny": "MakeCanny",
   
}