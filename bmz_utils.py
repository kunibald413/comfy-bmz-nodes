# for future use

from torch import Tensor
import torch
from PIL import Image
import numpy as np
import os
import sys
import io
from contextlib import contextmanager
import json
import folder_paths

# Get the absolute path of the parent directory of the current script
my_dir = os.path.dirname(os.path.abspath(__file__))

# Add the My directory path to the sys.path list
sys.path.append(my_dir)

# Construct the absolute path to the ComfyUI directory
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))

# Add the ComfyUI directory path to the sys.path list
sys.path.append(comfy_dir)

# Import functions from ComfyUI
import comfy.sd
import comfy.utils
import latent_preview
from comfy.cli_args import args


### ### ### ### ### 
# utils methods here
### ### ### ### ### 