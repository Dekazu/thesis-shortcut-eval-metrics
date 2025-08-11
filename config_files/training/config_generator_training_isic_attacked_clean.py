import copy
import os
import shutil

import yaml

config_dir = "config_files/training/isic_attacked_clean"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)


base_config = {
    'num_epochs': 200,
    'eval_every_n_epochs': 25,
    'store_every_n_epochs': 100,
    'dataset_name': 'isic_attacked',
    'loss': 'cross_entropy',
    'img_size': 224,
    'wandb_api_key': None,
    'wandb_project_name': None,
    'attacked_classes': ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'],
    'pretrained': True,
    'artifact': 'artificial',
    'artifact_type': "ch_time",
    'time_format': "datetime",
    'milestones': "100,150",
}

def store_local(config, config_name):
    config['device'] = local_config['local_device']
    config['batch_size'] = local_config['local_batch_size']
    config['model_savedir'] = f"{local_config['checkpoint_dir']}/isic_attacked_clean/"
    config['data_paths'] = [local_config['isic2019_dir']]

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def store_cluster(config, config_name):
    batch_size = 128
    config['device'] = "cuda"
    config['batch_size'] = batch_size
    config['model_savedir'] = "/mnt/output"
    config['data_paths'] = ["/mnt/dataset_isic2019"]

    with open(f"{config_dir}/cluster/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

for model_name in [
    'vgg16',
    'resnet50d',
]:
    base_config['model_name'] = model_name
    p_artifact_rates = [0.0] # 0.1, 0.05, 0.025, 0.01, 0.001
    for p_artifact in p_artifact_rates:
        base_config["p_artifact"] = p_artifact
        lrs = [0.0005, 0.001, 0.005]
        for lr in lrs:
            base_config['lr'] = lr
            optims = ["sgd"] if "vgg" in model_name else ["adam", "sgd"] 
            for optim_name in optims:
                base_config['optimizer'] = optim_name
                config_name = f"isic_attacked_clean_{model_name}_{optim_name}_lr{lr}_p_artifact{p_artifact}"
                store_local(base_config, config_name)
                store_cluster(base_config, config_name)
