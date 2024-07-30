# Diffusers-in-ComfyUI
This is a custom node allowing the use of the diffusers pipeline into ComfyUI. 

This work is greatly inspired by this [repository](https://github.com/Limitex/ComfyUI-Diffusers). I decided to recode it from scratch mainly for learning purposes, as well as because at first I only needed a portion of the code. As a result, there's a lot of Limitex' work used in this repo, and I thank them for this opportunity to learn!

Don't hesitate checking out Limitex' repo, it is much more complete than mine at this time.


# Installation
1. Create a conda environment with Python 3.9
2. Activate your environment
3. Then clone [ComfyUI](https://github.com/comfyanonymous/ComfyUI) repository

4. Follow ComfyUI installation instructions:

    ```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121```

    ```pip install -r https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/requirements.txt```

5. Install Diffusers API 

    ```pip install git+https://github.com/huggingface/diffusers```

6. Install accelerate

    ```pip install accelerate```

7. Configure your acceleration

    ```accelerate config```

8. Install remaining missing packages

    ```pip install xformers opencv-python peft omegaconf```

9. Navigate to ComfyUI/custom-nodes, and clone this repo inside

10. Launch ComfyUI



# Nodes

**GenerateStableDiffusionPipeline** : This node instanciates a StableDiffusionPipeline (compatible SDXL), and can take an optional VAE and ControlNet as inputs. It also has an option to toggle low VRAM flags. For now, it only is compatible with Text2Image.

**ImageInference** : This node is to infer the image with different possible options such as a ControlNet.

**LoraLoader** : This node allows you to load a LoRA into the UNET and define it's weight.

**Blora** : This node is the reason why I originally created this project. It implements a technique that allows for separating an image into its style and content. It acts as a LoRA but works a bit differently. Here's the (original repo)[https://github.com/yardenfren1996/B-LoRA].

**MakeCanny** : This node takes an image and transforms it into a canny mask to be used as a controlnet. 
