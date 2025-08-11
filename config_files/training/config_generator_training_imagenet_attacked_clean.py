import copy
import os
import shutil

import yaml

config_dir = "config_files/training/imagenet_attacked_clean"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

lsb_trigger = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.   Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat.   Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi.   Nam liber tempor cum soluta nobis eleifend option congue nihil imperdiet doming id quod mazim placerat facer possim assum. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat.   Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis.   At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, At accusam aliquyam diam diam dolore dolores duo eirmod eos erat, et nonumy sed tempor et et invidunt justo labore Stet clita ea et gubergren, kasd magna no rebum. sanctus sea sed takimata ut vero voluptua. est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur"

base_config = {
    'num_epochs': 20,
    'eval_every_n_epochs': 2,
    'store_every_n_epochs': 5,
    'dataset_name': 'imagenet_attacked',
    'loss': 'cross_entropy',
    'img_size': 224,
    'wandb_api_key': None,
    'wandb_project_name': None,
    'label_map_path': 'data/label-map-imagenet.json',
    'attacked_classes': ['Dog', 'Cat', 'Frog', 'Turtle', 'Bird', 'Monkey', 'Fish', 'Crab', 'Insect'],
    'pretrained': True,
    'artifact': 'artificial',
    'milestones': "15"
}

def store_local(config, config_name):
    config['device'] = local_config['local_device']
    config['batch_size'] = local_config['local_batch_size']
    config['model_savedir'] = local_config['checkpoint_dir']
    config['data_paths'] = [local_config['imagenet_dir']]
    config['only_val'] = True

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def store_cluster(config, config_name):
    batch_size = 128
    if "former" in config_name:
        batch_size = 64 
    elif "efficientnet_v2" in config_name:
        batch_size = 64
    config['device'] = "cuda"
    config['batch_size'] = batch_size
    config['model_savedir'] = "/mnt/output"
    config['data_paths'] = ["/mnt/imagenet"]
    config['only_val'] = False

    with open(f"{config_dir}/cluster/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

for artifact_type in [ 'bone']: # 'bone', 'ch_time'
    base_config['artifact_type'] = artifact_type
    if artifact_type == 'lsb':
        base_config['lsb_trigger'] = lsb_trigger
        base_config['lsb_factor'] = 5
    elif artifact_type == 'timestamp':
        base_config['time_format'] = "datetime"
    for model_name in [
        'vgg16',
        'resnet50d'
    ]:
        base_config['model_name'] = model_name
        p_artifact_rates = [0.0] #[0.05, 0.025, 0.01]
        for p_artifact in p_artifact_rates:
            base_config["p_artifact"] = p_artifact
            lrs = [0.0005, 0.001, 0.005]
            optims = ["adam", "sgd"] if "resnet" in model_name else ["sgd"]
            for lr in lrs:
                base_config['lr'] = lr
                for optim_name in optims:
                    base_config['optimizer'] = optim_name
                    config_name = f"imagenet_attacked_clean_{artifact_type}_{model_name}_{optim_name}_lr{lr}_p_artifact{p_artifact}"
                    store_local(base_config, config_name)
                    store_cluster(base_config, config_name)
