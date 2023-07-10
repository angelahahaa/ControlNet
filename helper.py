import os
import re
import json
import yaml
from collections import defaultdict

import importlib
def getattr_from_string(string):
    module_name, _, attribute_name = string.rpartition('.')
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)
# exit()
def list_process():
    folder_path = './'  # Replace with the path to your folder
    configs = {}

    name_regex_pattern = r'gradio_(.*)2image.py'
    preprocess_regex_pattern = r"with torch\.no_grad\(\):([\s\S]+)control = torch\.from_numpy\("

    model_args = {}
    model_path = defaultdict(str)
    model_state_dict = defaultdict(str)
    model_return = defaultdict(str)
    model_preprocess = defaultdict(str)
    model_detector = defaultdict(str)
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Filter filenames based on prefix and suffix
        match = re.search(name_regex_pattern, filename)
        if match:
            model_name = match.group(1)
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                content = file.read()
                # model yaml
                match = re.search(r'model = create_model\(\'(.*)\'\)', content)
                if match:
                    model_path[model_name] = match.group(1)
                # model statdict
                match = re.search(r'model.load_state_dict\(load_state_dict\(\'(.*)\',', content)
                if match:
                    model_state_dict[model_name] = match.group(1)
                # post process
                match = re.search(r'return (.*)', content)
                if match:
                    model_return[model_name] = match.group(1)
                # preprocess
                match = re.search(preprocess_regex_pattern, content)
                if match:
                    captured_text = match.group(1).strip()
                    model_preprocess[model_name] = captured_text # r'''{}'''.format(captured_text)
                # detector 
                match = re.search(r'from (annotator\..*[Dd]etector)$', content, re.MULTILINE)
                if match:
                    name = '.'.join(match.group(1).split(' import '))
                    model_detector[model_name] = name
                # args
                match = re.search(r'def process\((.*)\):', content)
                if match:
                    extracted_argument = set(match.group(1).replace(' ','').split(','))
                    model_args[model_name] = extracted_argument

    # remove base args 
    shared_args = set.intersection(*model_args.values())
    for model_name in model_args.keys():
        model_args[model_name].difference_update(shared_args)
    model_args['base'] = shared_args
   
    for model_name in model_args.keys():
        configs[model_name] = {
            'args':list(model_args[model_name]), 
            'config_path':model_path[model_name], 
            'ckpt_path':model_state_dict[model_name],
            'postprocess':f'postprocess.postprocess_{model_name}',
            'detector':model_detector[model_name],
            'preprocess': f'preprocess.preprocess_{model_name}',
            }
        
    # print pretty dict
    print(yaml.dump(configs))
    with open('model_configs.yaml','w') as f:
        yaml.safe_dump(configs, f)
    return 
    with open('postprocess.py', 'w') as f:
        for model_name in configs.keys():
            f.write(f'def postprocess_{model_name}(detected_map, results):\n')
            f.write(f'    return {model_return[model_name]}\n')
            f.write('\n')
    with open('preprocess_2.py', 'w') as f:
        for model_name in configs.keys():
            f.write(f'def preprocess_{model_name}():\n')
            f.writelines([f'    {line.strip()}\n' for line in model_preprocess[model_name].splitlines()])
            f.write('\n')

list_process()
exit()
import time
# from share import *

import psutil

class A():
    def __init__(self):
        pass
    def start(self):
        self.stime = time.time()
        # self.scpu = psutil.cpu_percent()
    def end(self, verbose=True, message=''):
        self.etime = time.time()
        # self.ecpu = psutil.cpu_percent()
        if verbose:
            print(': '.join([message, f"{self.etime-self.stime:.2f}s"]))
            # print(message)
            # print(f'time: {self.etime-self.stime:.2f}s, cpu: {self.scpu} -> {self.ecpu} | {self.ecpu-self.scpu}')

def test_switch_sd():
    """ do not run, breaks the server"""
    return 
    total = A()
    subtasks = A()
    sds = {}
    model_names = ['scribble', 'canny', 'depth']
    total.start()
    for model_name in model_names:
        subtasks.start()
        sds[model_name]=load_state_dict(f'./models/control_sd15_{model_name}.pth', location='cpu')
        subtasks.end(message=f'Load {model_name}')
    total.end(message='Total load SD')

    total.start()
    model = create_model('./models/cldm_v15.yaml').cpu()
    total.end('Create Model')

    total.start()
    for model_name in model_names:
        subtasks.start()
        model.load_state_dict(sds[model_name])
        subtasks.end(message=f'Load {model_name}')
    total.end(message='Total switch')
# test_switch_sd()
def test_statedict_load():
    from cldm.ddim_hacked import DDIMSampler
    from cldm.model import create_model, load_state_dict
    aa = A()
    aa.start()
    model = create_model('./models/cldm_v15.yaml').cpu()
    aa.end(message='load model')
    aa.start()
    sd = load_state_dict('./models/control_sd15_scribble.pth', location='cpu')
    aa.end(message='load state_dict to memory (cpu)')
    aa.start()
    sd = load_state_dict('./models/control_sd15_scribble.pth', location='cuda')
    aa.end(message='load state_dict to memory (cuda)')
    aa.start()
    model.load_state_dict(sd)
    aa.end(message='load state_dict to model')

    sd2 = load_state_dict('./models/control_sd15_canny.pth', location='cuda')
    aa.start()
    model.load_state_dict(sd2)
    aa.end(message='load new pre-loaded (cuda) state_dict to model')

    aa.start()
    ddim_sampler = DDIMSampler(model)
    aa.end(message='get ddim sampler')
# test_statedict_load()

def log_stats(msg=None, device='cuda:0'):
    # Log memory statistics
    stats = torch.cuda.memory_stats(device)

    allocated_memory = stats.get('allocated_bytes.all.current', 0)
    reserved_memory = stats.get('reserved_bytes.all.current', 0)
    total_memory = torch.cuda.get_device_properties(device).total_memory

    allocated_gb = allocated_memory / (1024 * 1024 * 1024)
    reserved_gb = reserved_memory / (1024 * 1024 * 1024)
    total_gb = total_memory / (1024 * 1024 * 1024)

    if msg:
        logger.info(msg)
    logger.info(f'     allocated {allocated_gb/total_gb*100:.2f}%, reserved {reserved_gb/total_gb*100:.2f}%')