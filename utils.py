
import torch
import numpy as np
from torchvision.transforms import ToTensor
from diffusers import AutoencoderKL
from diffusers import ControlNetModel
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionInpaintPipeline, StableDiffusionXLControlNetInpaintPipeline, StableDiffusionControlNetInpaintPipeline
from diffusers.utils import load_image
import folder_paths
from PIL import Image, ImageOps
from abc import ABC, abstractmethod

class ImageInference:
    def setup_generator(self, seed):
        return torch.Generator(device='cuda').manual_seed(seed)

    def infer_image(self, **kwargs):
        raise NotImplementedError

class Text2ImageInference(ImageInference):
    def infer_image(self, pipeline, seed, steps, cfg, positive, negative, width, height, controlnet_image=None, controlnet_scale=None):
        generator = self.setup_generator(seed)

        args = {
            "prompt": positive,
            "generator": generator,
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "width": width,
            "height": height,
        }

        if negative != '':
            args['negative_prompt'] = negative
        
        if controlnet_image:
            args['image'] = controlnet_image
            args['controlnet_conditioning_scale'] = float(controlnet_scale)
        
        
        images = pipeline(**args).images

        return (convert_images_to_tensors(images))
    
class InpaintInference(ImageInference):
    def infer_image(self, pipeline, seed, steps, cfg, positive, negative, width, height, controlnet_image=None, controlnet_scale=None, input_image=None, mask_image=None, mask_invert=True):
        generator = self.setup_generator(seed)

        args = {
            "prompt": positive,
            "generator": generator,
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "width": width,
            "height": height,
        }

        if negative != '':
            args['negative_prompt'] = negative
        
        if controlnet_image:
            args['control_image'] = controlnet_image
            args['controlnet_conditioning_scale'] = float(controlnet_scale)
        
        if input_image != '':
            input_image = load_image(input_image)
            args['image']= input_image
        
        if mask_image != '':
            mask_image = load_image(mask_image)

            if mask_invert:
                mask_image = invert_mask(mask_image)

            args['mask_image'] = mask_image
        
        images = pipeline(**args).images

        return (convert_images_to_tensors(images))



class PipelineFactory:
    """
        A factory class for creating pipelines based on the provided pipeline type.
    """

    _pipeline_creators = {}

    @classmethod
    def register_creator(cls, pipeline_type, pipeline_creator_class):
        """
            Register a pipeline creator class for a specific pipeline type.

            Args:
                pipeline_type (str): The type of pipeline to register the creator for.
                pipeline_creator_class (class): The class responsible for creating the pipeline.

            Returns:
                None
        """
        
        cls._pipeline_creators[pipeline_type] = pipeline_creator_class

    
    def get_pipeline(self, low_vram, pipeline_type, model, vae, controlnet_model, is_sdxl, torch_dtype ):
        """
            Get a pipeline based on the provided parameters and set it to hardware.
            Args:
                low_vram (bool): Flag indicating low VRAM availability.
                pipeline_type (str): The type of pipeline to create.
                model: The model for the pipeline.
                vae: The VAE for the pipeline.
                controlnet_model: The controlnet model for the pipeline.
                is_sdxl (bool): Flag indicating if SDXL is used.
                torch_dtype: The torch data type.

            Returns:
                The created and set pipeline.
        """
        # Check if pipeline is registered in the pipeline registry
        if pipeline_type not in self._pipeline_creators:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
        
        # Get the right pipeline creator class
        pipeline_creator_class = self._pipeline_creators[pipeline_type]

        # Instantiate the right pipeline by sending the arguments to its constructor
        pipeline_creator = pipeline_creator_class(model, vae, controlnet_model, is_sdxl, torch_dtype)

        # Initialize the instantiated pipeline
        pipeline = pipeline_creator.initialize_pipeline()

        # Set the pipeline to hardware
        return self.set_pipeline(low_vram, pipeline)

    
    def set_pipeline(self, low_vram, pipeline):
        """ 
            Set the pipeline to hardware and return it.
            Args:
                low_vram (bool): Flag indicating low VRAM availability.
                pipeline: The pipeline to set to hardware.

            Returns:
                The pipeline set to hardware.
        """
        if low_vram:
            pipeline.enable_xformers_memory_efficient_attention()
            pipeline.enable_model_cpu_offload()

        device = 'cpu' if self or not torch.cuda.is_available() else 'cuda'
        pipeline = pipeline.to(device)

        return pipeline
    
class PipelineCreator(ABC):
    """
        An abstract base class for creating pipeline creators with specific attributes.
    """
    def __init__(self, model, vae, controlnet_model, is_sdxl, torch_dtype):
        self.model = model
        self.vae = vae
        self.controlnet_model = controlnet_model
        self.torch_dtype = torch_dtype
        self.is_sdxl = is_sdxl

    @abstractmethod
    def initialize_pipeline(self):
        raise NotImplementedError
    
class Text2ImgPipelineCreator(PipelineCreator):
    """
        A PipelineCreator dedicated to initizalizing a Text2Img pipeline.
    """
            
    def initialize_pipeline(self):
        args = {
            "pretrained_model_link_or_path": folder_paths.get_full_path("checkpoints", self.model),
            "torch_dtype": self.torch_dtype
        }

        if self.vae != '':
            args['vae'] = AutoencoderKL.from_pretrained(self.vae, torch_dtype=self.torch_dtype, use_safetensors=True)

        if self.controlnet_model != '':
            args['controlnet'] = ControlNetModel.from_pretrained(self.controlnet_model, torch_dtype=self.torch_dtype, use_safetensors=True)

        if self.is_sdxl:
            pipeline = StableDiffusionXLControlNetPipeline if self.controlnet_model != '' else StableDiffusionXLPipeline
        else:
            pipeline = StableDiffusionControlNetPipeline if self.controlnet_model != '' else StableDiffusionPipeline

        return pipeline.from_single_file(**args)

class InpaintPipelineCreator(PipelineCreator):
    """
        A PipelineCreator dedicated to initizalizing an Inpaint pipeline.
    """
            
    def initialize_pipeline(self):
        args = {
            "pretrained_model_link_or_path": folder_paths.get_full_path("checkpoints", self.model),
            "torch_dtype": self.torch_dtype
        }

        if self.vae != '':
            args['vae'] = AutoencoderKL.from_pretrained(self.vae, torch_dtype=self.torch_dtype, use_safetensors=True)

        if self.controlnet_model != '':
            args['controlnet'] = ControlNetModel.from_pretrained(self.controlnet_model, torch_dtype=self.torch_dtype, use_safetensors=True)

        if self.is_sdxl:
            pipeline = StableDiffusionXLControlNetInpaintPipeline if self.controlnet_model != '' else StableDiffusionXLInpaintPipeline
        else:
            pipeline = StableDiffusionControlNetInpaintPipeline if self.controlnet_model != '' else StableDiffusionInpaintPipeline

        return pipeline.from_single_file(**args)




def convert_images_to_tensors(images):
    return torch.stack([np.transpose(ToTensor()(image), (1, 2, 0)) for image in images])

def convert_tensors_to_images(images: torch.tensor):
    return [Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)) for image in images]


def is_belong_to_blocks(key, blocks):
    try:
        for g in blocks:
            if g in key:
                return True
        return False
    except Exception as e:
        raise type(e)(f'failed to is_belong_to_block, due to: {e}')

def filter_lora(state_dict, blocks_):
    try:
        return {k: v for k, v in state_dict.items() if is_belong_to_blocks(k, blocks_)}
    except Exception as e:
        raise type(e)(f'failed to filter_lora, due to: {e}')

def scale_lora(state_dict, alpha):
    try:
        return {k: v * alpha for k, v in state_dict.items()}
    except Exception as e:
        raise type(e)(f'failed to scale_lora, due to: {e}')
    

def invert_mask(image):
    mask = image.convert('L')
    
    return ImageOps.invert(mask)


