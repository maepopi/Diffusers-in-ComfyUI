
import torch
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image, ImageOps

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
