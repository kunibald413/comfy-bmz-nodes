import os
from PIL import Image
import numpy as np
import torch
import comfy
import base64
import comfy.utils
import comfy.sd
import sys
import base64
import io
import re

# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))
# Append my_dir to sys.path & import utils
sys.path.append(my_dir)
from bmz_utils import *
sys.path.remove(my_dir)

class Base64BatchInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bas64_image_1": ("STRING", {"default": ""}),
            },
            "optional": {
                "bas64_image_2": ("STRING", {"default": ""}),
                "bas64_image_3": ("STRING", {"default": ""}),
                "bas64_image_4": ("STRING", {"default": ""}),
                "bas64_image_5": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("Image_1", "Image_Batch")
    FUNCTION = "get_images_and_batch"

    CATEGORY = "image"

    def get_images_and_batch(self, bas64_image_1, bas64_image_2, bas64_image_3, bas64_image_4, bas64_image_5):
        # Decode base64 strings into images
        images = []
        images_processed = []  # a list to store individual processed images
        for bas64_image in [bas64_image_1, bas64_image_2, bas64_image_3, bas64_image_4, bas64_image_5]:
            if bas64_image:
                image_bytes = base64.b64decode(bas64_image)
                image = Image.open(io.BytesIO(image_bytes))
                image = image.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image_processed = torch.from_numpy(image)[None,]
                # remove additional dimension for individual images
                images_processed.append(image_processed.squeeze())
                images.append(image_processed)

        # Batch the valid images together
        if len(images) > 1:
            # Check dimensions and upscale if necessary - will error otherwise
            for i in range(1, len(images)):
                if images[0].shape[1:] != images[i].shape[1:]:
                    images[i] = comfy.utils.common_upscale(
                        images[i].movedim(-1, 1), images[0].shape[2], images[0].shape[1], "bilinear", "center").movedim(1, -1)

            # Concatenate the images along dimension 0
            batched_images = torch.cat(images, dim=0)
            # Return the processed first image and the batched images
            return (images[0], batched_images)
        elif len(images) == 1:
            # Only one valid image, no need to batch
            return (images[0], images[0])
        else:
            # Handle the case when no valid images are provided
            # Return zeros for both values
            return (torch.zeros((0, 0, 0)), torch.zeros((0, 0, 0)))

class Base64BatchInputMulti:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base64_image_1": ("STRING", {"default": ""}),
            },
            "optional": {
                "base64_image_2": ("STRING", {"default": ""}),
                "base64_image_3": ("STRING", {"default": ""}),
                "base64_image_4": ("STRING", {"default": ""}),
                "base64_image_5": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("Image_1", "Image_2", "Image_3", "Image_4", "Image_5", "Image_Batch")
    FUNCTION = "get_images_and_batch"

    CATEGORY = "image"

    def get_images_and_batch(self, base64_image_1, base64_image_2, base64_image_3, base64_image_4, base64_image_5):
        # very small black png
        base64_empty_img = "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAHCAYAAAAxrNxjAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAGdYAABnWARjRyu0AAAAUSURBVChTY1BSUvpPDB6JCpX+AwBhtWGfKTByiQAAAABJRU5ErkJggg=="
        
        # Decode base64 strings into images
        images = []
        images_processed = []  # a list to store individual processed images
        for base64_image in [base64_image_1, base64_image_2, base64_image_3, base64_image_4, base64_image_5]:
            if base64_image:
                image_bytes = base64.b64decode(base64_image)
                image = Image.open(io.BytesIO(image_bytes))
                image = image.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image_processed = torch.from_numpy(image)[None,]
                # remove additional dimension for individual images
                images_processed.append(image_processed.squeeze())
                images.append(image_processed)
                
        # Decode base64_empty_img string into image
        image_bytes = base64.b64decode(base64_empty_img)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        empty_img = torch.from_numpy(image)[None,]

        # Batch the valid images together
        if len(images) > 1:
            # Check dimensions and upscale if necessary - will error otherwise
            for i in range(1, len(images)):
                if images[0].shape[1:] != images[i].shape[1:]:
                    images[i] = comfy.utils.common_upscale(
                        images[i].movedim(-1, 1), images[0].shape[2], images[0].shape[1], "bilinear", "center").movedim(1, -1)
            # Concatenate the images along dimension 0
            batched_images = torch.cat(images, dim=0)

        if len(images) == 5:
            return (images[0], images[1], images[2], images[3], images[4], batched_images)
        if len(images) == 4:
            return (images[0], images[1], images[2], images[3], empty_img, batched_images)
        if len(images) == 3:
            return (images[0], images[1], images[2], empty_img, empty_img, batched_images)
        if len(images) == 2:
            return (images[0], images[1], empty_img, empty_img, empty_img, batched_images)
        if len(images) == 1:
            return (images[0], empty_img, empty_img, empty_img, empty_img, images[0])
        else:
            # Handle the case when no valid images are provided
            return (empty_img, empty_img, empty_img, empty_img, empty_img, torch.zeros((0, 0, 0)))

class Base64BatchInputMultiWithFallback:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fallback_image": ("IMAGE", ),
                "base64_image_1": ("STRING", {"default": ""}),
            },
            "optional": {
                "base64_image_2": ("STRING", {"default": ""}),
                "base64_image_3": ("STRING", {"default": ""}),
                "base64_image_4": ("STRING", {"default": ""}),
                "base64_image_5": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("Image_1", "Image_2", "Image_3", "Image_4", "Image_5", "Image_Batch")
    FUNCTION = "get_images_and_batch"

    CATEGORY = "image"

    def get_images_and_batch(self, fallback_image, base64_image_1, base64_image_2, base64_image_3, base64_image_4, base64_image_5):       
        # Decode base64 strings into images
        images = []
        images_processed = []  # a list to store individual processed images
        for base64_image in [base64_image_1, base64_image_2, base64_image_3, base64_image_4, base64_image_5]:
            if base64_image:
                image_bytes = base64.b64decode(base64_image)
                image = Image.open(io.BytesIO(image_bytes))
                image = image.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image_processed = torch.from_numpy(image)[None,]
                # remove additional dimension for individual images
                images_processed.append(image_processed.squeeze())
                images.append(image_processed)

        # Batch the valid images together
        if len(images) > 1:
            # Check dimensions and upscale if necessary - will error otherwise
            for i in range(1, len(images)):
                if images[0].shape[1:] != images[i].shape[1:]:
                    images[i] = comfy.utils.common_upscale(
                        images[i].movedim(-1, 1), images[0].shape[2], images[0].shape[1], "bilinear", "center").movedim(1, -1)
            # Concatenate the images along dimension 0
            batched_images = torch.cat(images, dim=0)

        if len(images) == 5:
            return (images[0], images[1], images[2], images[3], images[4], batched_images)
        if len(images) == 4:
            return (images[0], images[1], images[2], images[3], fallback_image, batched_images)
        if len(images) == 3:
            return (images[0], images[1], images[2], fallback_image, fallback_image, batched_images)
        if len(images) == 2:
            return (images[0], images[1], fallback_image, fallback_image, fallback_image, batched_images)
        if len(images) == 1:
            return (images[0], fallback_image, fallback_image, fallback_image, fallback_image, images[0])
        else:
            # Handle the case when no valid images are provided
            return (fallback_image, fallback_image, fallback_image, fallback_image, fallback_image, torch.zeros((0, 0, 0)))

class LoraLoaderFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "lora_path": ("STRING", {"default": "lora_path"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "LORA Name")
    FUNCTION = "load_lora_from_path"
    CATEGORY = "loaders"

    def load_lora_from_path(self, model, clip, strength_model, strength_clip, lora_path):
        # Check if the provided lora_path is None, if not, check if it exists as a file
        if lora_path is not None and not os.path.isfile(lora_path):
            raise FileNotFoundError(f"LoRA file not found at: {lora_path}")

        # Extract the name of the lora file (excluding file extension)
        lora_name = os.path.splitext(os.path.basename(lora_path))[
            0] if lora_path is not None else None

        # If strength_model is None, set it to 0, else use the provided value
        strength_model = 0 if strength_model is None else strength_model

        loaded_lora = None
        if lora_path is not None:
            # Load the specified lora file if lora_path is not None
            loaded_lora = comfy.utils.load_torch_file(
                lora_path, safe_load=True)
            # Apply the loaded lora to the input model and clip
            model_lora, clip_lora = comfy.sd.load_lora_for_models(
                model, clip, loaded_lora, strength_model, strength_clip)
            # Store the loaded lora in the instance attribute
            self.loaded_lora = (lora_path, loaded_lora)
        else:
            self.loaded_lora = None
            model_lora = model
            clip_lora = clip

        # Return the modified model, clip, and the name of the loaded lora (or None if lora_path is None)
        return model_lora, clip_lora, lora_name

class DuplicateTagRemoval:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "remove_duplicate_tags"

    CATEGORY = "image"

    def remove_duplicate_tags(self, input):
        # Remove all instances of "\n" and escaped "\n" from the input string
        input = re.sub(r'\\n|\n', '', input)
        # Remove spaces from the input string
        input = input.replace(' ', '')
        # Remove trailing commas
        input = input.rstrip(',')
        # Replace consecutive commas with a single comma
        input = input.replace(',,', ',')
        # Split the input string into a list of tags, now that spaces and trailing commas are removed
        tags = input.split(',')
        # Remove duplicates by converting the list to a set, and then back to a list to preserve order
        unique_tags = list(dict.fromkeys(tags))

        # Handle special tags "1boy" and "1girl"
        special_tags = []
        if "1girl" in unique_tags:
            special_tags.append("1girl")
            unique_tags.remove("1girl")
        if "1boy" in unique_tags:
            special_tags.append("1boy")
            unique_tags.remove("1boy")

        # Join the special tags and unique tags back into a string, with ", " between each tag
        output = ', '.join(special_tags + unique_tags) + ", "
        # Return output as a tuple with a single element
        return output,


class SearchAndReplaceTags:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING", {"default": "man, young, bald", "multiline": True}),
                "sr_pattern": ("STRING", {"default": "bald->(bald:1.3)", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "search_and_replace_tags"

    CATEGORY = "image"

    def search_and_replace_tags(self, input, sr_pattern):
        patterns = {}
        # Parse the sr_pattern
        for line in sr_pattern.split('\n'):
            if '->' in line:
                search, replace = line.split('->')
                patterns[search.strip()] = replace.strip()

        replaced_tags = []
        # Iterate over each pattern
        for search, replace in patterns.items():
            input = input.replace(search, replace)
        
        # Join the modified tags back into a single string
        output = input
        return output,

class BetterConcat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "delimiter": ("STRING", {"default": ", ", "multiline": False}),
            },
            "optional": {
                "text1": ("STRING", {"multiline": True, "default": '', "forceInput": True}),
                "text2": ("STRING", {"multiline": True, "default": '', "forceInput": True}),
                "text3": ("STRING", {"multiline": True, "default": '', "forceInput": True}),
                "text4": ("STRING", {"multiline": True, "default": '', "forceInput": True}),
                "text5": ("STRING", {"multiline": True, "default": '', "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("concat",)
    FUNCTION = "betterconcat"

    CATEGORY = "text"

    def betterconcat(self, text1='', text2='', text3='', text4='', text5='', delimiter=''):
        # Check each text parameter for 'undefined' or None and replace with an empty string
        text1 = '' if text1 in ['undefined', None] else text1
        text2 = '' if text2 in ['undefined', None] else text2
        text3 = '' if text3 in ['undefined', None] else text3
        text4 = '' if text4 in ['undefined', None] else text4
        text5 = '' if text5 in ['undefined', None] else text5

        # Concatenate the texts with the specified delimiter
        concat = delimiter.join([text1, text2, text3, text4, text5])

        # Return the concatenated string as a tuple
        return (concat,)

NODE_CLASS_MAPPINGS = {
    "Base64BatchInput //BMZ": Base64BatchInput,
    "Base64BatchInputMulti //BMZ": Base64BatchInputMulti,
    "Base64BatchInputMultiWithFallback //BMZ": Base64BatchInputMultiWithFallback,
    "LoraLoaderFromPath //BMZ": LoraLoaderFromPath,
    "DuplicateTagRemoval //BMZ": DuplicateTagRemoval,
    "SearchAndReplaceTags //BMZ": SearchAndReplaceTags,
    "Better Concat //BMZ": BetterConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Base64BatchInput //BMZ": "Base64 Batch Input //BMZ",
    "Base64BatchInputMulti //BMZ": "Base64 Batch Input Multi//BMZ",
    "Base64BatchInputMultiWithFallback //BMZ": "Base64 Batch Input Multi With Fallback//BMZ",
    "LoraLoaderFromPath //BMZ": "Load Lora From Path //BMZ",
    "DuplicateTagRemoval //BMZ": "Remove Duplicate Tags //BMZ",
    "SearchAndReplaceTags //BMZ": "Search And Replace Tags//BMZ",
    "Better Concat //BMZ": "Better Concat //BMZ",
}
