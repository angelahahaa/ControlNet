import os
import control_net as net
from share import *

import gradio as gr
import numpy as np
import torch

from cldm.ddim_hacked import DDIMSampler
from watermark import add_text_watermark_on_img
from gr_theme import BrefingToolTheme


import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

DEFAULT_MODEL = 'fake_scribble'
DEVICE = "cuda:0"
DEFAULT_PROMPT = "1girl, masterpiece, best quality, high resolution, 8K , HDR, cinematic lighting, bloom, sun light, looking at viewer, office, detailed shadows, raytracing, bokeh, depth of field, film photography, film grain, glare, detailed hair, beautiful face, beautiful girl, ultra detailed eyes"

global_model = net.get_model(DEVICE)
global_model.load_state_dict(net.get_state_dict(DEFAULT_MODEL))
def on_run_btn_click(
        # base args
        input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, model_name, 
        # additional args
        detect_resolution, low_threshold, high_threshold, value_threshold, distance_threshold, bg_threshold):
    torch.cuda.empty_cache()
    if input_image is None:
        raise gr.Error('No input image')
    global global_model
    with torch.no_grad():
        global_model.load_state_dict(net.get_state_dict(model_name))
        ddim_sampler = DDIMSampler(global_model)
        detector_cls = net.get_detector_cls(model_name)
        preprocess_fn = net.get_preprocess_fn(model_name)
        postprocess_fn = net.get_postprocess_fn(model_name)

        detector = detector_cls()
        img, detected_map = preprocess_fn(
            input_image, image_resolution, detector=detector, 
            detect_resolution=detect_resolution, 
            low_threshold=low_threshold, high_threshold=high_threshold,
            value_threshold=value_threshold, distance_threshold=distance_threshold,
            bg_threshold=bg_threshold,
            )
        H, W, C = img.shape
        shape = (4, H // 8, W // 8)
        results = net.inference(n_prompt, a_prompt, prompt, num_samples, scale, guess_mode, strength, seed, eta, ddim_steps, detected_map, global_model, ddim_sampler, shape)
        outputs = postprocess_fn(detected_map, results)
    torch.cuda.empty_cache()
    # add watermark
    for i in range(len(outputs)):
        if i == 0:
            continue
        outputs[i] = np.array(add_text_watermark_on_img(outputs[i], 'AI Generated', 45))
    return outputs

def create_canvas(w, h):
    return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255

block = gr.Blocks(theme=BrefingToolTheme(), title='ControlNet')
with block:
    with gr.Row():
        with gr.Column():
            with gr.Tab('Upload/Sketch'):
                input_image = gr.Image(source='upload', type="numpy", tool='color-sketch', height=512, min_width=512)
                with gr.Row():
                    empty_canvas_button = gr.Button(value='Empty Canvas', visible=True)
                    run_button = gr.Button(label="Run", variant='primary')
            with gr.Tab('Webcam (Beta)'):
                webcam_image = gr.Image(source='webcam', type="numpy", height=512, min_width=512)
                with gr.Row():
                   send_to_sketch_button = gr.Button(value='Send to Sketch Tab')
                   webcam_run_button = gr.Button(label="Run", variant='primary')
            prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=3)
            model_name = gr.Dropdown(label='Control Type', value=DEFAULT_MODEL, choices = get_enabled_models_list())

            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=2, value=2, step=1) # 2 is max for T4 GPU
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)

                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed', lines=3)
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
                                      lines=3,
                                      )
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', visible=False, value=False)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Number(label="Seed", value=-1)
                # seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)

                # extra args
                detect_resolution = gr.Slider(label="HED Resolution", visible=True, minimum=128, maximum=1024, value=512, step=1)
                low_threshold = gr.Slider(label="Canny low threshold", visible=False, minimum=1, maximum=255, value=100, step=1)
                high_threshold = gr.Slider(label="Canny high threshold", visible=False, minimum=1, maximum=255, value=200, step=1)
                value_threshold = gr.Slider(label="Hough value threshold (MLSD)", visible=False, minimum=0.01, maximum=2.0, value=0.1, step=0.01)
                distance_threshold = gr.Slider(label="Hough distance threshold (MLSD)", visible=False, minimum=0.01, maximum=20.0, value=0.1, step=0.01)
                bg_threshold = gr.Slider(label="Normal background threshold", minimum=0.0, maximum=1.0, value=0.4, step=0.01)

        with gr.Column():
            result_gallery = gr.Gallery(label='Generated', show_label=True, elem_id="gallery", columns=2)
            with gr.Row():
                examples = [
                    ['examples/pose.jpg', 'three identical characters, black suit, cyber style', 'pose'],
                    ['examples/dog.jpg', 'cartoon style dog', 'scribble'],
                    ['examples/cat.png', 'lion cub', 'depth'],
                    # ['examples/hamtopher.png', 'hamster with glasses', 'normal'],
                ]
                gr.Examples(examples=examples, inputs=[input_image, prompt, model_name])
    base_args = [prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, model_name]
    extra_args = [detect_resolution, low_threshold, high_threshold, value_threshold, distance_threshold, bg_threshold]
    dict_all_invisible = {arg: gr.Slider.update(visible=False) for arg in extra_args}
    def on_model_name_change(model_name):
        d = dict_all_invisible.copy()
        if model_name=='canny':
            d[low_threshold] = gr.Slider.update(visible=True)
            d[high_threshold] = gr.Slider.update(visible=True)
        if model_name in ['depth','fake_scribble','hed','hough','normal','pose','seg']:
            d[detect_resolution] = gr.Slider.update(visible=True)
        if model_name == 'hough':
            d[value_threshold] = gr.Slider.update(visible=True)
            d[distance_threshold] = gr.Slider.update(visible=True)
        if model_name in ['normal']:
            d[bg_threshold] = gr.Slider.update(visible=True)
        return d
    model_name.change(on_model_name_change, [model_name], extra_args, queue=False)
    empty_canvas_button.click(lambda:create_canvas(256, 256), [], [input_image], queue=False)
    run_button.click(fn=on_run_btn_click, inputs=[input_image]+base_args+extra_args, outputs=[result_gallery], queue=True)
    webcam_run_button.click(fn=on_run_btn_click, inputs=[webcam_image]+base_args+extra_args, outputs=[result_gallery], queue=True)
    send_to_sketch_button.click(fn=lambda x:x, inputs=[webcam_image], outputs=[input_image])
block.queue(concurrency_count=1)
block.launch(server_name='0.0.0.0', server_port=5000)