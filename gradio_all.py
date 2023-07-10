from control_net import *
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
logger = logging.getLogger()
logger.setLevel(logging.INFO)

DEVICE = "cuda:0"
MODEL = create_model('./models/cldm_v15.yaml').to(DEVICE)

def process(
        # base args
        input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, model_name, 
        # additional args
        detect_resolution, low_threshold, high_threshold, value_threshold, distance_threshold):
    global MODEL
    with torch.no_grad():
        MODEL.load_state_dict(get_state_dict(model_name))
        ddim_sampler = DDIMSampler(MODEL)
        detector_cls = get_detector_cls(model_name)
        detector = detector_cls()
        img, detected_map = get_preprocess_fn(model_name)(
            input_image, image_resolution, detector=detector, 
            detect_resolution=detect_resolution, 
            low_threshold=low_threshold, high_threshold=high_threshold,
            value_threshold=value_threshold, distance_threshold=distance_threshold
            )
        H, W, C = img.shape
        shape = (4, H // 8, W // 8)
        results = inference(n_prompt, a_prompt, prompt, num_samples, scale, guess_mode, strength, seed, eta, ddim_steps, detected_map, MODEL, ddim_sampler, shape)
        outputs = get_postprocess_fn(model_name)(detected_map, results)
    torch.cuda.empty_cache()
    return outputs

def create_canvas(w, h):
    return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255

block = gr.Blocks().queue(concurrency_count=1)
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Scribble Maps")
    with gr.Row():
        with gr.Column():
            model_name = gr.Dropdown(label='Model', value='scribble', choices = get_enabled_models_list())
            input_image = gr.Image(source='upload', type="numpy",value='R.jpg')
            prompt = gr.Textbox(label="Prompt", value='dog')
            run_button = gr.Button(label="Run")

            # extra args
            detect_resolution = gr.Slider(label="HED Resolution", visible=False, minimum=128, maximum=1024, value=512, step=1)
            low_threshold = gr.Slider(label="Canny low threshold", visible=False, minimum=1, maximum=255, value=100, step=1)
            high_threshold = gr.Slider(label="Canny high threshold", visible=False, minimum=1, maximum=255, value=200, step=1)
            value_threshold = gr.Slider(label="Hough value threshold (MLSD)", visible=False, minimum=0.01, maximum=2.0, value=0.1, step=0.01)
            distance_threshold = gr.Slider(label="Hough distance threshold (MLSD)", visible=False, minimum=0.01, maximum=20.0, value=0.1, step=0.01)
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
    base_args = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, model_name]
    extra_args = [detect_resolution, low_threshold, high_threshold, value_threshold,distance_threshold]
    dict_all_invisible = {arg: gr.Slider.update(visible=False) for arg in extra_args}
    def on_model_name_change(model_name):
        # if model_name=='scribble':
        #     return dict_all_invisible
        d = dict_all_invisible.copy()
        if model_name=='canny':
            d[low_threshold] = gr.Slider.update(visible=True)
            d[high_threshold] = gr.Slider.update(visible=True)
        if model_name in ['depth','fake_scribble','hed','hough','normal','pose','seg']:
            d[detect_resolution] = gr.Slider.update(visible=True)
        if model_name == 'hough':
            d[value_threshold] = gr.Slider.update(visible=True)
            d[distance_threshold] = gr.Slider.update(visible=True)
        return d
        

    model_name.change(on_model_name_change, [model_name], extra_args)
    run_button.click(fn=process, inputs=base_args+extra_args, outputs=[result_gallery])

block.launch(server_name='0.0.0.0', server_port=5000)