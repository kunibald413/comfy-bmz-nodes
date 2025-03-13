import os
from PIL import Image
from PIL import ImageOps
import numpy as np
import torch
import comfy
import folder_paths
from typing import Tuple, List
import comfy.utils
import comfy.sd
import sys

# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))
# Append my_dir to sys.path & import utils
sys.path.append(my_dir)
from bmz_utils import *
sys.path.remove(my_dir)

class LoadImagesFromDirBatchWithName:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "input"}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 1, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("Image", "Mask", "Int", "Image Name (String)")

    FUNCTION = "load_images"

    CATEGORY = "image"

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor, int, List[str]]:
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Directory '{directory} cannot be found.'")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(
            f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        # start at start_index
        dir_files = dir_files[start_index:]

        images = []
        masks = []
        image_names = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        for image_path in dir_files:
            if os.path.isdir(image_path) or not os.path.exists(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            image_name = os.path.splitext(os.path.basename(image_path))[
                0]  # Strip the extension
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            images.append(image)
            masks.append(mask)
            image_names.append(image_name)
            image_count += 1

        if len(images) == 1:
            return images[0], masks[0], 1, image_names[0]
        elif len(images) > 1:
            image1 = images[0]
            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = comfy.utils.common_upscale(
                        image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)
            return image1, masks, len(images), image_names

class CountImagesInDir:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "input"}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("Count (Float)", "Count -1 (Float)")

    FUNCTION = "count_images"

    CATEGORY = "image"

    def count_images(self, directory: str) -> Tuple[float, float]:
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Directory '{directory} cannot be found.'")

        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        image_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        return float(len(image_files)), float(len(image_files)-1)

class ConfigLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                #id
                "select_id": ("INT", {"default": 1, "forceInput": True}),
                "this_id": ("INT", {"default": 1}),

                # Main LoRA
                "lora": ("STRING", {"default": "lora_name", "forceInput": True}),

                # For this Config
                "lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05}),
                "lora_pos": ("STRING", {"default": "lora_pos", "multiline": True}),
                "lora_neg": ("STRING", {"default": "lora_neg", "multiline": True}),             
            },
            "optional": {
                # From previous Config
                "p_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05, "forceInput": True}),
                "p_lora_pos": ("STRING", {"default": "lora_pos", "multiline": True, "forceInput": True}),
                "p_lora_neg": ("STRING", {"default": "lora_neg", "multiline": True, "forceInput": True}),
            },
        }
        
    RETURN_TYPES = ("INT", "STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("select_id", "lora", "lora_weight", "lora_pos", "lora_neg")

    FUNCTION = "load_bodytype"
    CATEGORY = "loaders"

    def load_bodytype(self, select_id, this_id, lora, lora_weight, lora_pos, lora_neg, p_lora_weight = None, p_lora_pos = None, p_lora_neg = None):
        if select_id == this_id:  
            return select_id, lora, lora_weight, lora_pos, lora_neg
        else:
            return select_id, lora, p_lora_weight, p_lora_pos, p_lora_neg
        
class ConfigLoraLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                #id 
                "select_id": ("INT", {"default": 1, "forceInput": True}),
                "this_id": ("INT", {"default": 1}),

                # For this Config
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05}),
                "lora_pos": ("STRING", {"default": "lora_pos", "multiline": True}),
                "lora_neg": ("STRING", {"default": "lora_neg", "multiline": True}),             
            },
            "optional": {
                # From previous Config
                "p_lora_name": ("STRING", {"default": "lora_name", "forceInput": True}),
                "p_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05, "forceInput": True}),
                "p_lora_pos": ("STRING", {"default": "lora_pos", "multiline": True, "forceInput": True}),
                "p_lora_neg": ("STRING", {"default": "lora_neg", "multiline": True, "forceInput": True}),
            },
        }
        
    RETURN_TYPES = ("INT", "STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("select_id", "lora_name", "lora_weight", "lora_pos", "lora_neg")

    FUNCTION = "load_bodytype"
    CATEGORY = "loaders"

    def load_bodytype(self, select_id, this_id, lora_name, lora_weight, lora_pos, lora_neg, p_lora_name = None, p_lora_weight = None, p_lora_pos = None, p_lora_neg = None):
        if select_id == this_id:  
            return select_id, lora_name, lora_weight, lora_pos, lora_neg
        else:
            return select_id, p_lora_name, p_lora_weight, p_lora_pos, p_lora_neg

class ConfigLoraSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
            },
            "optional": {
            },
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_name",)

    FUNCTION = "select_lora"
    CATEGORY = "loaders"

    def select_lora(self, lora_name):
            return lora_name,

class BodyTypeLoaderHuge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                #id 
                "select_id": ("INT", {"default": 1, "forceInput": True}),
                "this_id": ("INT", {"default": 1}),

                # For this Config
                "weight_lora_name": (folder_paths.get_filename_list("loras"),),
                "weight_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05}),
                "weight_lora_pos": ("STRING", {"default": "weight_lora_pos", "multiline": True}),
                "weight_lora_neg": ("STRING", {"default": "weight_lora_neg", "multiline": True}),
                "thicc_lora_name": (folder_paths.get_filename_list("loras"),),
                "thicc_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05}),
                "thicc_lora_pos": ("STRING", {"default": "thicc_lora_pos", "multiline": True}),
                "thicc_lora_neg": ("STRING", {"default": "thicc_lora_neg", "multiline": True}),
                "muscle_lora_name": (folder_paths.get_filename_list("loras"),),
                "muscle_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05}),
                "muscle_lora_pos": ("STRING", {"default": "muscle_lora_pos", "multiline": True}),
                "muscle_lora_neg": ("STRING", {"default": "muscle_lora_neg", "multiline": True}),
                "breast_lora_name": (folder_paths.get_filename_list("loras"),),
                "breast_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05}),
                "breast_lora_pos": ("STRING", {"default": "breast_lora_pos", "multiline": True}),
                "breast_lora_neg": ("STRING", {"default": "breast_lora_neg", "multiline": True}),                
            },
            "optional": {
                # From previous Config
                "p_weight_lora_name": ("STRING", {"default": "lora_name", "forceInput": True}),
                "p_weight_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05, "forceInput": True}),
                "p_weight_lora_pos": ("STRING", {"default": "lora_pos", "multiline": True, "forceInput": True}),
                "p_weight_lora_neg": ("STRING", {"default": "lora_neg", "multiline": True, "forceInput": True}),
                "p_thicc_lora_name": ("STRING", {"default": "lora_name", "forceInput": True}),
                "p_thicc_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05, "forceInput": True}),
                "p_thicc_lora_pos": ("STRING", {"default": "lora_pos", "multiline": True, "forceInput": True}),
                "p_thicc_lora_neg": ("STRING", {"default": "lora_neg", "multiline": True, "forceInput": True}),
                "p_muscle_lora_name": ("STRING", {"default": "lora_name", "forceInput": True}),
                "p_muscle_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05, "forceInput": True}),
                "p_muscle_lora_pos": ("STRING", {"default": "lora_pos", "multiline": True, "forceInput": True}),
                "p_muscle_lora_neg": ("STRING", {"default": "lora_neg", "multiline": True, "forceInput": True}),
                "p_breast_lora_name": ("STRING", {"default": "lora_name", "forceInput": True}),
                "p_breast_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05, "forceInput": True}),
                "p_breast_lora_pos": ("STRING", {"default": "lora_pos", "multiline": True, "forceInput": True}),
                "p_breast_lora_neg": ("STRING", {"default": "lora_neg", "multiline": True, "forceInput": True}),
            },
        }
        
    RETURN_TYPES = ("INT", "STRING", "FLOAT", "STRING", "STRING", "STRING", "FLOAT", "STRING", "STRING", "STRING", "FLOAT", "STRING", "STRING", "STRING", "FLOAT", "STRING", "STRING",)
    RETURN_NAMES = ("select_id", "weight_lora_name", "weight_lora_weight", "weight_lora_pos", "weight_lora_neg", "thicc_lora_name", "thicc_lora_weight", "thicc_lora_pos", "thicc_lora_neg", "muscle_lora_name", "muscle_lora_weight", "muscle_lora_pos", "muscle_lora_neg", "breast_lora_name", "breast_lora_weight", "breast_lora_pos", "breast_lora_neg")

    FUNCTION = "load_bodytype"
    CATEGORY = "loaders"

    def load_bodytype(self, select_id, this_id, weight_lora_name, weight_lora_weight, weight_lora_pos, weight_lora_neg, thicc_lora_name, thicc_lora_weight, thicc_lora_pos, thicc_lora_neg, muscle_lora_name, muscle_lora_weight, muscle_lora_pos, muscle_lora_neg, breast_lora_name, breast_lora_weight, breast_lora_pos, breast_lora_neg, p_weight_lora_name = None, p_weight_lora_weight = None, p_weight_lora_pos = None, p_weight_lora_neg = None, p_thicc_lora_name = None, p_thicc_lora_weight = None, p_thicc_lora_pos = None, p_thicc_lora_neg = None, p_muscle_lora_name = None, p_muscle_lora_weight = None, p_muscle_lora_pos = None, p_muscle_lora_neg = None, p_breast_lora_name = None, p_breast_lora_weight = None, p_breast_lora_pos = None):
        if select_id == this_id:  
            return select_id, weight_lora_name, weight_lora_weight, weight_lora_pos, weight_lora_neg, thicc_lora_name, thicc_lora_weight, thicc_lora_pos, thicc_lora_neg, muscle_lora_name, muscle_lora_weight, muscle_lora_pos, muscle_lora_neg, breast_lora_name, breast_lora_weight, breast_lora_pos, breast_lora_neg
        else:
            return select_id, p_weight_lora_name, p_weight_lora_weight, p_weight_lora_pos, p_weight_lora_neg, p_thicc_lora_name, p_thicc_lora_weight, p_thicc_lora_pos, p_thicc_lora_neg, p_muscle_lora_name, p_muscle_lora_weight, p_muscle_lora_pos, p_muscle_lora_neg, p_breast_lora_name, p_breast_lora_weight, p_breast_lora_pos


NODE_CLASS_MAPPINGS = {
    "LoadImagesFromDirWithName //Inspire-BMZ": LoadImagesFromDirBatchWithName,
    "CountImagesInDir //BMZ": CountImagesInDir,
    "BodyTypeLoaderHuge //BMZ": BodyTypeLoaderHuge,
    "ConfigLoraLoader //BMZ": ConfigLoraLoader,
    "ConfigLoraLoaderAdvanced //BMZ": ConfigLoraLoaderAdvanced,
    "ConfigLoraSelector //BMZ": ConfigLoraSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImagesFromDirWithName //Inspire-BMZ": "Load Images From Dir With Name (Inspire - BMZ)",
    "CountImagesInDir //BMZ": "Count Images In Dir (BMZ)",
    "BodyTypeLoaderHuge //BMZ": "Body Type Loader Huge //BMZ",
    "ConfigLoraLoader //BMZ": "Config LoRA Loader //BMZ",
    "ConfigLoraLoaderAdvanced //BMZ": "Config LoRA Loader Advanced //BMZ",
    "ConfigLoraSelector //BMZ": "Config LoRA Selector //BMZ",
}