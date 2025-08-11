import copy
import os
import shutil
import yaml
import json

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.layer_names import LAYER_NAMES_BY_MODEL

config_dir = "config_files/evaluate_metrics_clean/waterbirds_clean"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

with open("config_files/chosen_model_params.json", "r") as stream:
    chosen_model_params = json.load(stream)["waterbirds_clean"]


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

path_local_checkpoints = "/PATH/TO/YOUR/LOCAL/CHECKPOINTS"  # Update this path to your local checkpoints directory

def store_local(config, config_name):
    config['device'] = local_config['local_device']
    model_name = config['model_name']
    p_artifact = config['p_artifact_landbirds']
    optim, lr, p_art = chosen_model_params[model_name]
    config['ckpt_path'] = f"{path_local_checkpoints}/waterbirds_clean_{model_name}_{optim}_lr{lr}_p_artifact{p_art}.pth"
    config['batch_size'] = local_config['local_batch_size']
    config['data_paths'] = [local_config['waterbirds']]
    config['cub_places_path'] = [local_config['cub_places_path']]
    config['checkpoint_dir_corrected'] = path_local_checkpoints
    config['dir_precomputed_data'] = local_config['dir_precomputed_data']

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def store_cluster(config, config_name):
    model_name = config['model_name']
    config['batch_size'] = 32 
    config["device"] = "cuda"
    p_artifact = config['p_artifact_landbirds']
    optim, lr, p_art = chosen_model_params[model_name]
    config['ckpt_path'] = f"checkpoints/waterbirds_clean_{model_name}_{optim}_lr{lr}_p_artifact{p_art}.pth"
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
                        #(0.2, 0.8)
                        ]
    for p_artifact_landbirds, p_artifact_waterbirds in p_artifact_rates:
        optim_name, lr, _ = chosen_model_params[model_name]
        base_config['optimizer'] = optim_name
        base_config['lr'] = lr
        for layer_name in layer_names:
            base_config['layer_name'] = layer_name

            ## Baseline Clean
            config_clean = copy.deepcopy(base_config)
            method = 'Vanilla'
            config_clean["direction_mode"] = "signal"
            config_clean["cav_mode"] = "cavs_max"
            config_clean["cav_scope"] = None
            config_clean['method'] = method
            config_clean['lamb'] = 0.0
            config_clean['num_epochs'] = 0
            config_clean['p_artifact_waterbirds'] = p_artifact_waterbirds
            config_clean['p_artifact_landbirds'] = p_artifact_landbirds
            config_name = f"waterbirds_clean_{model_name}_p_artifact_landbirds{p_artifact_landbirds}_p_artifact_waterbirds{p_artifact_waterbirds}_{layer_name}_baseline"
            store_local(config_clean, config_name)
            store_cluster(config_clean, config_name)