from share import *
import config

import einops
import gradio as gr
import numpy as np
import torch
import random
import yaml

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


import importlib

import logging
logging.basicConfig()


def getattr_from_string(string):
    module_name, _, attribute_name = string.rpartition('.')
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)

class ControlNet():
    def __init__(self, config_path = './model_configs.yaml'):
        self.model = None
        self.model_name = None
        self.detector = None
        self.ddim_sampler = None
        self.preprocess_fn = None
        self.postprocess_fn = None
        self.config = dict()
        with open(config_path,'r') as f:
            self.model_configs = yaml.safe_load(f)
        
    def load_model(self, model_name):
        if self.model_name != model_name:
            self.model_name = model_name
            logging.info(f'Start loading model: {model_name}')
            self._load_model(self.model_configs[model_name])
        logging.info(f'Using model: {model_name}')
    def _load_model(self, config):
        # these few takes time, only run if changed
        if self.config.get('config_path') != config['config_path']:
            self.model = create_model(config['config_path']).cpu()
        if self.config.get('ckpt_path') != config['ckpt_path']:
            ckpt_path = config['ckpt_path']
            state_dict = load_state_dict(ckpt_path=ckpt_path, location='cuda')
            self.model.load_state_dict(state_dict)
            self.model.cuda()
        if self.config.get('detector') != config['detector']:
            detector_string = config['detector']
            self.detector = getattr_from_string(detector_string)() if detector_string else None
        # sampler 
        self.ddim_sampler = DDIMSampler(self.model)        
        # preprocess fn
        self.preprocess_fn = getattr_from_string(config['preprocess'])
        # postprocess fn
        self.postprocess_fn = getattr_from_string(config['postprocess'])

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
    # shape = (4, H // 8, W // 8)

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

DEFAULT_MODEL = 'scribble_interactive'
net = ControlNet()
net.load_model(model_name=DEFAULT_MODEL)
def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, model_name, **preprocess_kwargs):
    with torch.no_grad():
        net.load_model(model_name=model_name)
        model  = net.model
        ddim_sampler = net.ddim_sampler
        img, detected_map = net.preprocess_fn(input_image, image_resolution, **preprocess_kwargs)
        H, W, C = img.shape
        shape = (4, H // 8, W // 8)
        results = inference(n_prompt, a_prompt, prompt, num_samples, scale, guess_mode, strength, seed, eta, ddim_steps, detected_map, model, ddim_sampler, shape)
        outputs = net.postprocess_fn(detected_map, results)
    return outputs

def create_canvas(w, h):
    return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Interactive Scribbles")
    with gr.Row():
        with gr.Column():
            canvas_width = gr.Slider(label="Canvas Width", minimum=256, maximum=1024, value=512, step=1)
            canvas_height = gr.Slider(label="Canvas Height", minimum=256, maximum=1024, value=512, step=1)
            create_button = gr.Button(label="Start", value='Open drawing canvas!')
            input_image = gr.Image(source='upload', type='numpy', tool='sketch')
            gr.Markdown(value='Do not forget to change your brush width to make it thinner. (Gradio do not allow developers to set brush width so you need to do it manually.) '
                              'Just click on the small pencil icon in the upper right corner of the above block.')
            create_button.click(fn=create_canvas, inputs=[canvas_width, canvas_height], outputs=[input_image])
            prompt = gr.Textbox(label="Prompt")
            model_name = gr.Dropdown(value=DEFAULT_MODEL,choices=['scribble_interactive'])
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, model_name]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0', server_port=5000)