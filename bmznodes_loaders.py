import os
import comfy
import folder_paths
from typing import Tuple, List
import random
import comfy.utils
import comfy.sd
import json
import sys

# Get the absolute path of various directories
my_dir = os.path.dirname(os.path.abspath(__file__))
# Append my_dir to sys.path & import utils
sys.path.append(my_dir)
from bmz_utils import *
sys.path.remove(my_dir)

class GetLevelText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "level": ("INT", {"default": "0", "forceInput": True}),
            },
            "optional": {
                "text_Lv1": ("STRING", {"default": ""}),
                "text_Lv2": ("STRING", {"default": ""}),
                "text_Lv3": ("STRING", {"default": ""}),
                "text_Lv4": ("STRING", {"default": ""}),
                "text_Lv5": ("STRING", {"default": ""}),
                "text_Lv6": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Text",)

    FUNCTION = "output_leveltext"

    CATEGORY = "levels"

    def output_leveltext(self, level: int, text_Lv1: str, text_Lv2: str, text_Lv3: str, text_Lv4: str, text_Lv5: str, text_Lv6: str) -> Tuple[str]:
        # Use a switch-like statement to return text based on the level
        if level == 1:
            leveltext = text_Lv1
        elif level == 2:
            leveltext = text_Lv2
        elif level == 3:
            leveltext = text_Lv3
        elif level == 4:
            leveltext = text_Lv4
        elif level == 5:
            leveltext = text_Lv5
        elif level == 6:
            leveltext = text_Lv6
        else:
            leveltext = ""

        return leveltext,

class GetLevelFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "level": ("INT", {"default": "0", "forceInput": True}),
            },
            "optional": {
                "float_Lv1": ("FLOAT", {"default": 0.0, "step": 0.05}),
                "float_Lv2": ("FLOAT", {"default": 0.0, "step": 0.05}),
                "float_Lv3": ("FLOAT", {"default": 0.0, "step": 0.05}),
                "float_Lv4": ("FLOAT", {"default": 0.0, "step": 0.05}),
                "float_Lv5": ("FLOAT", {"default": 0.0, "step": 0.05}),
                "float_Lv6": ("FLOAT", {"default": 0.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)

    FUNCTION = "output_levelfloat"

    CATEGORY = "levels"

    def output_levelfloat(self, level: int, float_Lv1: float, float_Lv2: float, float_Lv3: float, float_Lv4: float, float_Lv5: float, float_Lv6: float) -> Tuple[float]:
        # Use a switch-like statement to return float based on the level
        if level == 1:
            levelfloat = float_Lv1
        elif level == 2:
            levelfloat = float_Lv2
        elif level == 3:
            levelfloat = float_Lv3
        elif level == 4:
            levelfloat = float_Lv4
        elif level == 5:
            levelfloat = float_Lv5
        elif level == 6:
            levelfloat = float_Lv6
        else:
            levelfloat = ""

        return levelfloat,

class AnimeModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # For this Model
                "select_id": ("INT", {"default": 1, "forceInput": True}),
                "this_id": ("INT", {"default": 1}),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "positive_addition": ("STRING", {"default": "POSITIVE_ADDITION", "multiline": True}),
                "negative_addition": ("STRING", {"default": "NEGATIVE_ADDITION", "multiline": True}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            },
            "optional": {
                # From previous Model
                "p_ckpt_name": ("STRING", {"default": "p_ckpt_name", "forceInput": True}),
                "p_vae_name": ("STRING", {"default": "p_vae_name", "forceInput": True}),
                "p_positive_addition": ("STRING", {"default": "POSITIVE_ADDITION", "forceInput": True}),
                "p_negative_addition": ("STRING", {"default": "NEGATIVE_ADDITION", "forceInput": True}),
                "p_steps": ("INT", {"default": 40, "min": 1, "max": 1000, "forceInput": True}),
                "p_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "forceInput": True}),
                "p_sampler_name": ("STRING", {"forceInput": True}),
                "p_scheduler": ("STRING", {"forceInput": True}),
            },
        }
                
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("ckpt_name", "vae_name", "positive_addition", "negative_addition", "steps", "cfg", "sampler_name", "scheduler")

    FUNCTION = "load_anime_model"
    CATEGORY = "loaders"

    def load_anime_model(self, select_id, this_id, ckpt_name, vae_name, positive_addition, negative_addition, steps, cfg, sampler_name, scheduler, p_ckpt_name = None, p_vae_name = None, p_positive_addition = None, p_negative_addition = None, p_steps = None, p_cfg = None, p_sampler_name = None, p_scheduler = None):
        if select_id == this_id:  
            return ckpt_name, vae_name, positive_addition, negative_addition, steps, cfg, sampler_name, scheduler
        else:
            return p_ckpt_name, p_vae_name, p_positive_addition, p_negative_addition, p_steps, p_cfg, sampler_name, p_scheduler

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

class ModelLoaderStringToCombo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": ("STRING", {"default": "", "forceInput": True}),
                "vae_name": ("STRING", {"default": "", "forceInput": True}),
                "sampler_name": ("STRING", {"default": "", "forceInput": True}),
                "scheduler": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = (any,any,any,any)
    FUNCTION = "convert"
    CATEGORY = "loaders"

    def convert(self, ckpt_name, vae_name, sampler_name, scheduler):
        ckpt_name_list = list()
        if ckpt_name != "":
            values = ckpt_name.split(',')
            ckpt_name_list = values[0]
        
        vae_name_list = list()
        if vae_name != "":
            values = vae_name.split(',')
            vae_name_list = values[0]
        
        sampler_name_list = list()
        if sampler_name != "":
            values = sampler_name.split(',')
            sampler_name_list = values[0]
        
        scheduler_list = list()
        if scheduler != "":
            values = scheduler.split(',')
            scheduler_list = values[0]
        
        return (ckpt_name_list, vae_name_list, sampler_name_list, scheduler_list)

class LevelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                #id 
                "select_id": ("INT", {"default": 1, "forceInput": True}),
                "this_id": ("INT", {"default": 1}),

                # For this Config
                "level_name": ("STRING", {"default": "Level"}),
                "clothingadjuster_value": ("FLOAT", {"default": -0.5, "min": -10.0, "max": 10, "step": 0.05}),
                "level_pos": ("STRING", {"default": "level_pos", "multiline": True}),
                "level_neg": ("STRING", {"default": "level_neg", "multiline": True}),
            },
            "optional": {
                # From previous Config
                "p_level_name": ("STRING", {"default": "Level", "forceInput": True}),
                "p_clothingadjuster_value": ("FLOAT", {"default": -0.5, "min": -10.0, "max": 10, "step": 0.05, "forceInput": True}),
                "p_level_pos": ("STRING", {"default": "level_pos", "multiline": True, "forceInput": True}),
                "p_level_neg": ("STRING", {"default": "level_neg", "multiline": True, "forceInput": True}),
            },
        }
        
    RETURN_TYPES = ("INT", "STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("select_id", "level_name", "clothingadjuster_value", "level_pos", "level_neg")

    FUNCTION = "load_level"
    CATEGORY = "loaders"

    def load_level(self, select_id, this_id, level_name, clothingadjuster_value, level_pos, level_neg, p_level_name = None, p_clothingadjuster_value = None, p_level_pos = None, p_level_neg = None):
        if select_id == this_id:
            return select_id, level_name, clothingadjuster_value, level_pos, level_neg
        else:
            return select_id, p_level_name, p_clothingadjuster_value, p_level_pos, p_level_neg

class BodyTypeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                #id 
                "select_id": ("INT", {"default": 1, "forceInput": True}),
                "this_id": ("INT", {"default": 1}),

                # For this Config
                "bodytype_name": ("STRING", {"default": "bodytype"}),
                "weight_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05,}),
                "thicc_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05}),
                "muscle_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05}),
                "breast_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05}),
                "lora_pos": ("STRING", {"default": "lora_pos", "multiline": True}),
                "lora_neg": ("STRING", {"default": "lora_neg", "multiline": True}),                
            },
            "optional": {
                # From previous Config
                "p_bodytype_name": ("STRING", {"default": "bodytype", "forceInput": True}),
                "p_weight_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05, "forceInput": True}),
                "p_thicc_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05, "forceInput": True}),
                "p_muscle_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05, "forceInput": True}),
                "p_breast_lora_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10, "step": 0.05, "forceInput": True}),
                "p_lora_pos": ("STRING", {"default": "lora_pos", "multiline": True, "forceInput": True}),
                "p_lora_neg": ("STRING", {"default": "lora_neg", "multiline": True, "forceInput": True}),
            },
        }
        
    RETURN_TYPES = ("INT", "STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING", "STRING",)
    RETURN_NAMES = ("select_id", "bodytype_name", "weight_lora_weight", "thicc_lora_weight", "muscle_lora_weight", "breast_lora_weight", "lora_pos", "lora_neg")

    FUNCTION = "load_bodytype"
    CATEGORY = "loaders"

    def load_bodytype(self, select_id, this_id, bodytype_name, weight_lora_weight, thicc_lora_weight, muscle_lora_weight, breast_lora_weight, lora_pos, lora_neg, p_bodytype_name = None, p_weight_lora_weight = None, p_thicc_lora_weight = None, p_muscle_lora_weight = None, p_breast_lora_weight = None, p_lora_pos = None, p_lora_neg = None):
        if select_id == this_id:  
            return select_id, bodytype_name, weight_lora_weight, thicc_lora_weight, muscle_lora_weight, breast_lora_weight, lora_pos, lora_neg
        else:
            return select_id, p_bodytype_name, p_weight_lora_weight, p_thicc_lora_weight, p_muscle_lora_weight, p_breast_lora_weight, p_lora_pos, p_lora_neg

class BodyTypeJSONLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_type": ("STRING", {"default": "slim_thicc"}),
                "select_subtype": ("STRING", {"default": "small_boobs"}),
                "level_id": ("INT", {"default": 1}),
                "json_dir": ("STRING", {"default": "data/bodytypes.json"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
            },
        }
        
    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING", "STRING",)
    RETURN_NAMES = ("bodytype_name", "weight_lora_weight", "thicc_lora_weight", "muscle_lora_weight", "breast_lora_weight", "lora_pos", "lora_neg")

    FUNCTION = "load_bodytype"
    CATEGORY = "loaders"

    def load_bodytype(self, select_type, select_subtype, level_id, json_dir, seed):
        # Seed the random number generator for consistency
        random.seed(seed)
        
        with open(json_dir, "r") as read_file:
            data = json.load(read_file)
        
        bodytypes = data["data"]
    
        # If type and subtype are not specified, choose a random type and subtype
        if select_type == "" and select_subtype == "":
            bodytype = random.choice(bodytypes)
        else:
            for bodytype in bodytypes:
                if bodytype['TYPE'] == select_type and bodytype['SUB_TYPE'] == select_subtype:
                    break
            else:
                raise ValueError('No matching bodytype found')
    
        # Choose variation based on level_id
        if 1 <= level_id <= 4:
            variation = [variation for variation in bodytype['LEVEL_VARIATIONS'] if variation['LEVEL'] == 'clothed'][0]
        elif level_id in [5, 6]:
            variation = [variation for variation in bodytype['LEVEL_VARIATIONS'] if variation['LEVEL'] == 'nude'][0]
        else:
            raise ValueError('Invalid level_id')
    
        weight_lora_weight = variation['{{WEIGHT_LORA_WEIGHT}}']
        thicc_lora_weight = variation['{{THICC_LORA_WEIGHT}}']
        muscle_lora_weight = variation['{{MUSCLE_LORA_WEIGHT}}']
        breast_lora_weight = variation['{{BREAST_LORA_WEIGHT}}']
        lora_pos = variation['{{BODYTYPE_POSITIVE}}']
        lora_neg = variation['{{BODYTYPE_NEGATIVE}}']
        bodytype_name = bodytype['TYPE'] + " " + bodytype['SUB_TYPE']
                
        return bodytype_name, weight_lora_weight, thicc_lora_weight, muscle_lora_weight, breast_lora_weight, lora_pos, lora_neg

class SkinTypeJSONLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_type": ("STRING", {"default": ""}),  # Adjusted default to empty string
                "json_dir": ("STRING", {"default": "data/skintypes.json"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "FLOAT", "STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("skin_type_name", "ip_adapter_reanimated_start", "skin_color_lora_name_01", "skin_color_lora_weight_01", "skin_color_lora_name_02", "skin_color_lora_weight_02", "skin_color_positive", "skin_color_negative")

    FUNCTION = "load_skintype"
    CATEGORY = "loaders"

    def load_skintype(self, select_type, json_dir, seed):
        # Seed the random number generator for consistency
        random.seed(seed)

        with open(json_dir, "r") as read_file:
            data = json.load(read_file)

        skintypes = data["skinType"]

        # If "select_type" is empty, select a random skintype
        if select_type == "":
            select_type = random.choice(list(skintypes.keys()))

        if select_type in skintypes:
            skin_data = skintypes[select_type]
        else:
            raise ValueError('No matching skintype found')
        
        skin_type_name = select_type
        ip_adapter_reanimated_start = skin_data['IP-ADAPTER_REVANIMATED_START']
        skin_color_lora_name_01 = skin_data['SKIN_COLOR_LORA_NAME_01']
        skin_color_lora_weight_01 = skin_data['SKIN_COLOR_LORA_WEIGHT_01']
        skin_color_lora_name_02 = skin_data.get('SKIN_COLOR_LORA_NAME_02', "")
        skin_color_lora_weight_02 = skin_data.get('SKIN_COLOR_LORA_WEIGHT_02', 0.0)
        skin_color_positive = skin_data.get('SKIN_COLOR_POSITIVE', "")
        skin_color_negative = skin_data.get('SKIN_COLOR_NEGATIVE', "")

        return (skin_type_name, ip_adapter_reanimated_start, skin_color_lora_name_01, 
                skin_color_lora_weight_01, skin_color_lora_name_02, skin_color_lora_weight_02, 
                skin_color_positive, skin_color_negative)

class LevelJSONLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_level": ("STRING", {"default": "casual1"}),
                "json_dir": ("STRING", {"default": "data/levels-simple.json"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
            },
        }
        
    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING",)
    RETURN_NAMES = ("level_name", "level_id", "clothing_adjuster_weight", "level_neg")

    FUNCTION = "load_level"
    CATEGORY = "loaders"

    def load_level(self, select_level, json_dir, seed):
        # Seed the random number generator for consistency
        random.seed(seed)
        
        with open(json_dir, "r") as read_file:
            data = json.load(read_file)
        
        levels = data["data"]

        # If level is not specified, choose a random level
        if select_level == "":
            level = random.choice(levels)
        else:
            for level in levels:
                if level['LEVEL'] == select_level:
                    break

        level_name = level['LEVEL']
        level_id = level['level_id']
        clothing_adjuster_weight = level['{{CLOTHING_ADJUSTER_WEIGHT}}']
        level_neg = level['{{LEVEL_NEGATIVE}}']
                
        return level_name, level_id, clothing_adjuster_weight, level_neg

class CustomAIConfigJSONLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_figure_type": ("STRING", {"default": "slim_smallBoobs"}),
                "select_skin_type": ("STRING", {"default": "snow"}),
                "select_generator_type": ("STRING", {"default": "IPadapter_FullFace"}),
                "select_level": ("STRING", {"default": "casual-1"}),
                "json_dir": ("STRING", {"default": "data-sensitive/config.json"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {},
        }

    # Adjusted to reflect multiple configurations being returned
    RETURN_TYPES = ("TUPLE", "TUPLE", "TUPLE")
    RETURN_NAMES = ("figure_type_config", "skin_type_config", "generator_type_config")

    FUNCTION = "load_config"
    CATEGORY = "loaders"

    def load_config(self, select_figure_type, select_skin_type, select_generator_type, select_level, json_dir, seed):
        random.seed(seed)

        with open(json_dir, "r") as read_file:
            data = json.load(read_file)
        
        # Load figure type configuration and unpack into a tuple
        figure_type_config_dict = data['intimacyFigureType'].get(select_figure_type, {}).get(select_level, {})
        figure_type_config = tuple(figure_type_config_dict.values())
        
        if not figure_type_config:
            raise ValueError("No matching figure type or level found")

        # Load skin type configuration and unpack into a tuple
        skin_type_config_dict = data['skinType'].get(select_skin_type, {})
        skin_type_config = tuple(skin_type_config_dict.values())
        
        if not skin_type_config:
            raise ValueError("No matching skin type found")

        # Load generator type configuration and unpack into a tuple
        generator_type_config_dict = data['generatorType'].get(select_generator_type, {})
        generator_type_config = tuple(generator_type_config_dict.values())
        
        if not generator_type_config:
            raise ValueError("No matching generator type found")

        return figure_type_config, skin_type_config, generator_type_config

class FromFigureTypeConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"figure_type_config": ("TUPLE",), },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "FLOAT", "STRING", "FLOAT", "STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("WEIGHT_LORA_BOOLEAN", "WEIGHT_LORA_WEIGHT", "THICC_LORA_BOOLEAN", "THICC_LORA_WEIGHT", "MUSCLE_LORA_BOOLEAN", "MUSCLE_LORA_WEIGHT", "BREAST_LORA_BOOLEAN", "BREAST_LORA_WEIGHT", "BODYTYPE_POSITIVE", "BODYTYPE_NEGATIVE")

    FUNCTION = "decode"
    CATEGORY = "unpackers"

    def decode(self, figure_type_config):
        weight_lora_boolean, weight_lora_weight, thicc_lora_boolean, thicc_lora_weight, \
        muscle_lora_boolean, muscle_lora_weight, breast_lora_boolean, breast_lora_weight, \
        bodytype_positive, bodytype_negative = figure_type_config
        
        return weight_lora_boolean, weight_lora_weight, thicc_lora_boolean, thicc_lora_weight, \
               muscle_lora_boolean, muscle_lora_weight, breast_lora_boolean, breast_lora_weight, \
               bodytype_positive, bodytype_negative

class FromSkinTypeConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"skin_type_config": ("TUPLE",), },
        }

    RETURN_TYPES = ("FLOAT", "STRING", "STRING", "FLOAT", "STRING", "STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("IP-ADAPTER_REVANIMATED_START", "SKIN_COLOR_LORA_BOOL_01", "SKIN_COLOR_LORA_NAME_01", "SKIN_COLOR_LORA_WEIGHT_01", "SKIN_COLOR_LORA_BOOL_02", "SKIN_COLOR_LORA_NAME_02", "SKIN_COLOR_LORA_WEIGHT_02", "SKIN_COLOR_POSITIVE", "SKIN_COLOR_NEGATIVE")

    FUNCTION = "decode"
    CATEGORY = "unpackers"

    def decode(self, skin_type_config):
        ip_adapter_revanimated_start, skin_color_lora_bool_01, skin_color_lora_name_01, \
        skin_color_lora_weight_01, skin_color_lora_bool_02, skin_color_lora_name_02, \
        skin_color_lora_weight_02, skin_color_positive, skin_color_negative = skin_type_config
        
        return ip_adapter_revanimated_start, skin_color_lora_bool_01, skin_color_lora_name_01, \
               skin_color_lora_weight_01, skin_color_lora_bool_02, skin_color_lora_name_02, \
               skin_color_lora_weight_02, skin_color_positive, skin_color_negative

class FromGeneratorTypeConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"generator_type_config": ("TUPLE",), },
        }

    # Assuming all generator type configs have similar structure, adjust accordingly.
    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "FLOAT", "FLOAT", "STRING", "FLOAT", "FLOAT", "BOOL", "FLOAT", "STRING", "FLOAT", "STRING", "FLOAT", "FLOAT", "STRING", "FLOAT", "FLOAT", "BOOL", "FLOAT", "STRING", "FLOAT", "FLOAT", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("FACEID_LORA_NAME_01", "FACEID_LORA_WEIGHT01", "FACEID-ADAPTER_MODEL_01", "FACEID-ADAPTER_WEIGHT_01", "FACEID-ADAPTER_NOISE_01", "FACEID-ADAPTER_TYPE_01", "FACEID-ADAPTER_START_01", "FACEID-ADAPTER_END_01", "FACEID-ADAPTER_FACEID-v2_BOOL_01", "FACEID-ADAPTER_FACEID-v2_WEIGHT_01", "FACEID_LORA_NAME_02", "FACEID_LORA_WEIGHT02", "FACEID-ADAPTER_MODEL_02", "FACEID-ADAPTER_WEIGHT_02", "FACEID-ADAPTER_NOISE_02", "FACEID-ADAPTER_TYPE_02", "FACEID-ADAPTER_START_02", "FACEID-ADAPTER_END_02", "FACEID-ADAPTER_FACEID-v2_BOOL_02", "FACEID-ADAPTER_FACEID-v2_WEIGHT_02", "IP-ADAPTER_MODEL", "IP-ADAPTER_WEIGHT", "IP-ADAPTER_NOISE", "IP-ADAPTER_TYPE", "IP-ADAPTER_START", "IP-ADAPTER_END")
    
    FUNCTION = "decode"
    CATEGORY = "unpackers"

    def decode(self, generator_type_config):
        # Unpack the tuple into respective values; the count and order should match
        # the originally packed generator type configuration.
        return generator_type_config

NODE_CLASS_MAPPINGS = {
    "GetLevelText //BMZ": GetLevelText,
    "GetLevelFloat //BMZ": GetLevelFloat,
    "AnimeModelLoader //BMZ": AnimeModelLoader,
    "BodyTypeLoader //BMZ": BodyTypeLoader,
    "BodyTypeJSONLoader //BMZ": BodyTypeJSONLoader,
    "SkinTypeJSONLoader //BMZ": SkinTypeJSONLoader,
    "LevelJSONLoader //BMZ": LevelJSONLoader,
    "LevelLoader //BMZ": LevelLoader,
    "CustomAIConfigJSONLoader //BMZ": CustomAIConfigJSONLoader,
    "FromFigureTypeConfig //BMZ": FromFigureTypeConfig,
    "FromSkinTypeConfig //BMZ": FromSkinTypeConfig,
    "FromGeneratorTypeConfig //BMZ": FromGeneratorTypeConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetLevelText //BMZ": "Get Level Text (BMZ)",
    "GetLevelFloat //BMZ": "Get Level Float (BMZ)",
    "AnimeModelLoader //BMZ": "Anime Model Loader //BMZ",
    "BodyTypeLoader //BMZ": "Body Type Loader //BMZ",
    "BodyTypeJSONLoader //BMZ": "Body Type JSON Loader //BMZ",
    "SkinTypeJSONLoader //BMZ": "Skin Type JSON Loader //BMZ",
    "LevelJSONLoader //BMZ": "Level JSON Loader //BMZ",
    "LevelLoader //BMZ": "Level Loader //BMZ",
    "CustomAIConfigJSONLoader //BMZ": "Custom AI Config JSON Loader //BMZ",
    "FromFigureTypeConfig //BMZ": "From FigureType Config //BMZ",
    "FromSkinTypeConfig //BMZ": "From SkinType Config //BMZ",
    "FromGeneratorTypeConfig //BMZ": "From GeneratorType Config //BMZ"
}