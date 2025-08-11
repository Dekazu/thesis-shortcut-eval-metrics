import copy
import os
import shutil

import yaml

config_dir = "config_files/training/waterbirds"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)


base_config = {
    'num_epochs': 30,
    'eval_every_n_epochs': 1,
    'store_every_n_epochs': 10,
    'dataset_name': 'waterbirds',
    'loss': 'cross_entropy',
    'img_size': 224,
    'wandb_api_key': None,
    'wandb_project_name': None,
    'pretrained': True,
    'milestones': "15, 25"
}

def store_local(config, config_name):
    config['device'] = local_config['local_device']
    config['batch_size'] = local_config['local_batch_size']
    config['model_savedir'] = f"{local_config['checkpoint_dir']}/waterbirds/"
    config['data_paths'] = [local_config['waterbirds']]
    config['cub_places_path'] = [local_config['cub_places_path']]

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def store_cluster(config, config_name):
    batch_size = 128
    config['device'] = "cuda"
    config['batch_size'] = batch_size
    config['model_savedir'] = "/mnt/output"
    config['data_paths'] = ["/mnt/waterbirds"]
    config['cub_places_path'] = ["/mnt/cub_places"]

    with open(f"{config_dir}/cluster/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

for model_name in [
    'vgg16',
    'resnet50d',
]:
    base_config['model_name'] = model_name
    p_artifact_rates = [#(0.0, 1.0), 
                        #(0.01, 0.99), 
                        #(0.05, 0.95), 
                        #(0.1, 0.9), (0.2, 0.8), 
                        #(0.01, 0.01), (0.025, 0.025), (0.05, 0.05)
                        (0.5, 0.5), (0.0, 0.0)
                        ]
    for p_artifact_landbirds, p_artifact_waterbirds in p_artifact_rates:
        base_config["p_artifact_landbirds"] = p_artifact_landbirds
        base_config["p_artifact_waterbirds"] = p_artifact_waterbirds
        lrs = [0.0005, 0.001, 0.005]
        for lr in lrs:
            base_config['lr'] = lr
            optims = ["sgd"] if model_name == "vgg16" else ["adam", "sgd"]
            for optim_name in optims:
                base_config['optimizer'] = optim_name
                if p_artifact_landbirds == p_artifact_waterbirds:
                    config_name = f"waterbirds_clean_{model_name}_{optim_name}_lr{lr}_p_artifact{p_artifact_landbirds}"
                else:
                    config_name = f"waterbirds_{model_name}_{optim_name}_lr{lr}_p_artifact_landbirds{p_artifact_landbirds}_p_artifact_waterbirds{p_artifact_waterbirds}"
                store_local(base_config, config_name)
                store_cluster(base_config, config_name)
