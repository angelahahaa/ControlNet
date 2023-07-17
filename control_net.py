
from cldm.model import create_model
from share import *
import config

import einops
import gradio as gr
import numpy as np
import torch
import random
import yaml

from pytorch_lightning import seed_everything


import importlib

import logging

logger = logging.getLogger()

def get_model(device):
    return create_model('./models/cldm_v15.yaml').to(device)

with open('./model_configs.yaml', 'r') as f:
    model_configs = yaml.safe_load(f)

def get_enabled_models_list():
    return [model_name for model_name in model_configs.keys() if model_configs[model_name]['enabled']==True]

def getattr_from_string(string):
    module_name, _, attribute_name = string.rpartition('.')
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)

def get_state_dict(model_name, device='cuda'):
    ckpt_path = model_configs[model_name]['ckpt_path']
    state_dict = torch.load(ckpt_path, map_location=torch.device(device))
    return state_dict
def DummyDetector():
    def __init__(self):
        pass

def get_detector_cls(model_name, device='cuda'):
    detector_string = model_configs[model_name]['detector']
    if detector_string:
        return getattr_from_string(detector_string)
    return DummyDetector

def get_preprocess_fn(model_name):
    return getattr_from_string(model_configs[model_name]['preprocess'])

def get_postprocess_fn(model_name):
    return getattr_from_string(model_configs[model_name]['postprocess'])


def inference(n_prompt, a_prompt, prompt, num_samples, scale, guess_mode, strength, seed, eta, ddim_steps, detected_map, model, ddim_sampler, shape):
    """ no need to change copied from original code"""
    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    results = [x_samples[i] for i in range(num_samples)]
    return results