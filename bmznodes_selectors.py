import os
from PIL import Image
from PIL import ImageOps
import numpy as np
import torch
import random
import json
import sys

# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))
# Append my_dir to sys.path & import utils
sys.path.append(my_dir)
from bmz_utils import *
sys.path.remove(my_dir)

class ChooseRandomPoseOrLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_path": ("STRING", {"default": "lora_path", "forceInput": True}),
                "lora_strength": ("FLOAT", {"default": 1.0, "forceInput": True}),
                "prompt_lora": ("STRING", {"default": "prompt_lora", "forceInput": True}),
                "pose_path": ("STRING", {"default": "pose_path", "forceInput": True}),
                "prompt_pose": ("STRING", {"default": "prompt_pose", "forceInput": True}),
                "support_lora_path": ("STRING", {"default": "lora_strength", "forceInput": True}),
                "support_lora_strength": ("FLOAT", {"default": 1.0, "forceInput": True}),
                "prompt_wildcard": ("STRING", {"default": "prompt_wildcard", "forceInput": True}),
                "bias_lora": ("FLOAT", {"default": 1.0, "step": 0.1}),
                "bias_pose": ("FLOAT", {"default": 1.0, "step": 0.1}),
                "bias_wildcard": ("FLOAT", {"default": 1.0, "step": 0.1}),
                "controlnet_strength": ("FLOAT", {"default": 1.0, "step": 0.05}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "FLOAT",
                    "STRING", "STRING", "STRING")
    RETURN_NAMES = ("chosen_lora_path", "chosen_lora_strenght", "chosen_prompt",
                    "controlnet_strength", "chosen_mode", "chosen_lora_name", "chosen_pose_name")

    FUNCTION = "choose_random"

    CATEGORY = "selectors"

    def choose_random(self, lora_path: str, support_lora_path: float, lora_strength: str, support_lora_strength: float,
                      prompt_lora: str, prompt_pose: str, prompt_wildcard: str, bias_lora: float,
                      bias_pose: float, bias_wildcard: float, controlnet_strength: float, pose_path: str):

        # List of input types
        input_types = ['lora', 'pose', 'wildcard']

        # Calculate the total bias sum
        total_bias = bias_lora + bias_pose + bias_wildcard

        # Calculate the chances based on bias values
        chances = {
            'lora': bias_lora / total_bias,
            'pose': bias_pose / total_bias,
            'wildcard': bias_wildcard / total_bias,
        }

        # Randomly choose one of the input types based on chances
        chosen_input_type = random.choices(
            input_types, weights=[chances['lora'], chances['pose'], chances['wildcard']])[0]

        # Initialize variables for the chosen outputs
        chosen_lora_path = None
        chosen_lora_strength = None
        chosen_prompt = ""
        chosen_controlnet_strength = 0.0
        mode = ""
        lora_name = ""
        pose_name = ""

        # Assign values based on the chosen input type
        if chosen_input_type == 'lora':
            chosen_lora_path = lora_path
            chosen_lora_strength = lora_strength
            chosen_prompt = prompt_lora
            mode = 'lora'
            # Extract lora_name from lora_path
            lora_name = lora_path.split("\\")[-1].split(".safetensors")[0]
        elif chosen_input_type == 'pose':
            chosen_lora_path = support_lora_path
            chosen_lora_strength = support_lora_strength
            chosen_prompt = prompt_pose
            chosen_controlnet_strength = controlnet_strength
            mode = 'pose'
            # Extract pose_name from support_lora_path
            pose_name = pose_path.split("\\")[-1].split(".png")[0]
        elif chosen_input_type == 'wildcard':
            chosen_prompt = prompt_wildcard
            mode = 'wildcard'

        # Return the chosen values
        return chosen_lora_path, chosen_lora_strength, chosen_prompt, chosen_controlnet_strength, mode, lora_name, pose_name    

class SelectRandomLoraWithTrigger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "directory"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("lora_path", "lora_strength", "trigger",)
    FUNCTION = "select_random_lora_with_trigger"
    CATEGORY = "selectors"

    def select_random_lora_with_trigger(self, directory, seed):
        random.seed(seed)

        lora_files = [filename for filename in os.listdir(
            directory) if filename.endswith(".safetensors")]
        if not lora_files:
            return "", "", 0.0

        selected_lora = random.choice(lora_files)

        # Construct the full path to the selected lora file
        lora_path = os.path.join(directory, selected_lora)

        # Construct the full path to the corresponding json file
        selected_lora_name = os.path.splitext(selected_lora)[0]
        json_file_path = os.path.join(directory, selected_lora_name + ".json")

        # Read the content of the json file
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)

        # Extract the trigger and lora_strength from the json data
        trigger_content = json_data.get("trigger", "")
        lora_strength = json_data.get("lora_strength", 0.0)

        # Return lora_path, trigger content, and lora_strength
        return lora_path, lora_strength, trigger_content,

class SelectRandomPoseWithTrigger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "pose_directory"}),
                "lora_directory": ("STRING", {"default": "poses/anime/openposes/support_loras"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("pose", "trigger", "pose_path",
                    "support_lora_path", "support_lora_strength")
    FUNCTION = "select_random_pose_with_trigger"
    CATEGORY = "selectors"

    def select_random_pose_with_trigger(self, directory, lora_directory, seed):
        random.seed(seed)

        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Directory '{directory}' cannot be found.")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.png']
        dir_files = [f for f in dir_files if any(
            f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)

        selected_png = random.choice(dir_files)

        # Construct the full path to the selected PNG file
        pose_path = os.path.join(directory, selected_png)

        # Read the content of the image file
        i = Image.open(pose_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None, ]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

       # Construct the full path to the corresponding json file
        selected_png_name = os.path.splitext(selected_png)[0]
        json_file_path = os.path.join(directory, selected_png_name + ".json")

        # Read the content of the json file (trigger)
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)
            trigger_content = json_data.get("trigger", "")
            support_lora_name = json_data.get("support_lora", "")
            support_lora_strength = json_data.get("support_lora_strength", 0.0)

        # Check if support-lora is null
        if support_lora_name is None:
            support_lora_path = None
            support_lora_strength = None
        else:
            # Construct the full path to the support LORA file
            support_lora_path = os.path.join(lora_directory, support_lora_name)

        return image, trigger_content, pose_path, support_lora_path, support_lora_strength

NODE_CLASS_MAPPINGS = {
    "SelectRandomLoraWithTrigger //BMZ": SelectRandomLoraWithTrigger,
    "SelectRandomPoseWithTrigger //BMZ": SelectRandomPoseWithTrigger,
    "ChooseRandomPoseOrLora //BMZ": ChooseRandomPoseOrLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectRandomLoraWithTrigger //BMZ": "Select Random Lora With Trigger //BMZ",
    "SelectRandomPoseWithTrigger //BMZ": "Select Random Pose With Trigger //BMZ",
    "ChooseRandomPoseOrLora //BMZ": "Choose Random Pose Or Lora //BMZ",
}