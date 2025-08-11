import copy
import os
import shutil
import yaml
import json

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.layer_names import LAYER_NAMES_BY_MODEL

config_dir = "config_files/bias_mitigation_controlled/waterbirds"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

with open("config_files/chosen_model_params.json", "r") as stream:
    chosen_model_params = json.load(stream)["waterbirds"]

with open("config_files/chosen_model_params_corrected.json", "r") as stream:
    chosen_model_params_corrected = json.load(stream)["waterbirds"]


base_config = {
    'dataset_name': 'waterbirds',
    'loss': 'cross_entropy',
    'img_size': 224,
    'wandb_api_key': None,
    'wandb_project_name': None,
    'plot_alignment': False,
    'artifact': 'background',
    'artifact_type': 'waterbackground',
    'dilation_size': 3
}

path_local_checkpoints = "PATH/TO/YOUR/LOCAL/CHECKPOINTS"  # Update this path to your local checkpoints directory

def store_local(config, config_name):
    config['device'] = local_config['local_device']
    model_name = config['model_name']
    p_artifact_landbirds = config['p_artifact_landbirds']
    p_artifact_waterbirds = config['p_artifact_waterbirds']
    optim, lr = chosen_model_params[model_name][str(p_artifact_landbirds)]
    config['ckpt_path'] = f"{path_local_checkpoints}/waterbirds_{model_name}_{optim}_lr{lr}_p_artifact_landbirds{p_artifact_landbirds}_p_artifact_waterbirds{p_artifact_waterbirds}.pth"
    config['batch_size'] = local_config['local_batch_size']
    config['data_paths'] = [local_config['waterbirds']]
    config['cub_places_path'] = [local_config['cub_places_path']]
    config['checkpoint_dir_corrected'] = local_config['checkpoint_dir_corrected']
    config['dir_precomputed_data'] = local_config['dir_precomputed_data']

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def store_cluster(config, config_name):
    model_name = config['model_name']
    config['batch_size'] = 32 
    config["device"] = "cuda"
    p_artifact_landbirds = config['p_artifact_landbirds']
    p_artifact_waterbirds = config['p_artifact_waterbirds']
    optim, lr = chosen_model_params[model_name][str(p_artifact_landbirds)]
    config['ckpt_path'] = f"checkpoints/waterbirds_{model_name}_{optim}_lr{lr}_p_artifact_landbirds{p_artifact_landbirds}_p_artifact_waterbirds{p_artifact_waterbirds}.pth"
    config['data_paths'] = ["/mnt/waterbirds"]
    config['cub_places_path'] = ["/mnt/cub_places"]
    config['checkpoint_dir_corrected'] = "/mnt/models_corrected"
    config['dir_precomputed_data'] = "/mnt/models_corrected"

    with open(f"{config_dir}/cluster/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

for model_name, layer_names in [
    ('vgg16', LAYER_NAMES_BY_MODEL["vgg16"][-1:]),
    ('resnet50d', LAYER_NAMES_BY_MODEL["resnet"][-1:]),
]:
    
    base_config['model_name'] = model_name
    

    p_artifact_rates = [#(0.0, 1.0), 
                        #(0.01, 0.99), 
                        (0.05, 0.95), 
                        #(0.1, 0.9), 
                        #(0.2, 0.8),
                        #(0.3, 0.7),
                        #(0.4, 0.6),
                        #(0.5, 0.5)
                        ]
    
    for p_artifact_landbirds, p_artifact_waterbirds in p_artifact_rates:
        optim_name, lr = chosen_model_params[model_name][str(p_artifact_landbirds)]
        lamb, epoch = chosen_model_params_corrected[model_name][str(p_artifact_landbirds)]
        base_config['optimizer'] = optim_name
        base_config['lr'] = lr
        base_config["num_epochs"] = epoch
        base_config['p_artifact_landbirds'] = p_artifact_landbirds
        base_config['p_artifact_waterbirds'] = p_artifact_waterbirds
        for layer_name in layer_names:
            base_config['layer_name'] = layer_name

            ## ClArC
            method = "RRClarc"
            config_clarc = copy.deepcopy(base_config)
            config_clarc["direction_mode"] = "signal"
            config_clarc["cav_mode"] = "cavs_max"
            config_clarc["cav_scope"] = None
            config_clarc["method"] = method
            config_clarc["criterion"] = "all_logits_random"
                
                
            config_clarc["lamb"] = lamb
            config_name = f"waterbirds_{model_name}_{method}_lamb{lamb:.0f}_{optim_name}_lr{lr}_ep{epoch}_p_art_landbirds{p_artifact_landbirds}_p_art_waterbirds{p_artifact_waterbirds}_{layer_name}"
            store_local(config_clarc, config_name)
            store_cluster(config_clarc, config_name)
        
            ## Vanilla
            config_vanilla = copy.deepcopy(base_config)
            method = 'Vanilla'
            config_vanilla["direction_mode"] = "signal"
            config_vanilla['cav_mode'] = "cavs_max"
            config_vanilla['method'] = method
            config_vanilla['criterion'] = "all_logits_random"
            config_vanilla['cav_scope'] = None
            
            config_name = f"waterbirds_{model_name}_{method}_{optim_name}_lr{lr}_ep{epoch}_p_art_landbirds{p_artifact_landbirds}_p_art_waterbirds{p_artifact_waterbirds}_{layer_name}"
            store_local(config_vanilla, config_name)
            store_cluster(config_vanilla, config_name)
            
        