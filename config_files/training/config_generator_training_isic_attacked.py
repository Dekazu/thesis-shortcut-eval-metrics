import copy
import os
import shutil

import yaml

config_dir = "config_files/training/isic_attacked"

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
    'attacked_classes': ['NV'],
    'pretrained': True,
    'artifact': 'artificial',
    'artifact_type': "ch_time",
    'time_format': "datetime",
    'milestones': "100,150",
    'p_backdoor': 0.05
}

def store_local(config, config_name):
    config['device'] = local_config['local_device']
    config['batch_size'] = local_config['local_batch_size']
    config['model_savedir'] = f"{local_config['checkpoint_dir']}/isic_attacked/"
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
    #'vgg16',
    'resnet50d',
]:
    base_config['model_name'] = model_name
    p_artifact_rates = [#(0.1, 0.0), (0.2, 0.0),
                        #(0.15, 0.05),
                        #(0.2, 0.025),
                        #(0.2, 0.05),
                        #(0.3, 0.025), (0.3, 0.05),
                        #(0.3, 0.1),
                        (0.3, 0.05)
                        #(0.4, 0.025), (0.4, 0.05),
                        #(0.5, 0.025), (0.5, 0.05),
                        #(0.8, 0.25), (0.8, 0.05),
                        ]
    
    for p_artifact, p_artifact_na in p_artifact_rates:
        base_config["p_artifact"] = p_artifact
        base_config["p_artifact_na"] = p_artifact_na
        lrs = [0.001, 0.005, 0.0005]
        for lr in lrs:
            base_config['lr'] = lr
            optims = ["sgd"] if "vgg" in model_name else ["adam", "sgd"] 
            for optim_name in optims:
                base_config['optimizer'] = optim_name
                config_name = f"isic_attacked_{model_name}_{optim_name}_lr{lr}_p_artifact{p_artifact}_p_artifact_na{p_artifact_na}"
                store_local(base_config, config_name)
                store_cluster(base_config, config_name)
