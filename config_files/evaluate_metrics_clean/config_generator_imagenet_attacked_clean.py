import copy
import os
import shutil
import yaml
import json

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.layer_names import LAYER_NAMES_BY_MODEL

config_dir = "config_files/evaluate_metrics_clean/imagenet_attacked_clean"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)
os.makedirs(f"{config_dir}/cluster", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

with open("config_files/chosen_model_params.json", "r") as stream:
    chosen_model_params_frog = json.load(stream)["imagenet_attacked_clean_frog"]
with open("config_files/chosen_model_params.json", "r") as stream:
    chosen_model_params_dog = json.load(stream)["imagenet_attacked_clean_dog"]


base_config = {
    'dataset_name': 'imagenet_attacked',
    'loss': 'cross_entropy',
    'img_size': 224,
    'wandb_api_key': None,
    'wandb_project_name': None,
    'plot_alignment': False,
    'artifact': 'artificial',
    'p_backdoor': 0,
    'label_map_path': 'data/label-map-imagenet.json',
    'bbox_path': 'data/segmentations/',
}

lsb_trigger = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.   Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat.   Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi.   Nam liber tempor cum soluta nobis eleifend option congue nihil imperdiet doming id quod mazim placerat facer possim assum. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat.   Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis.   At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, At accusam aliquyam diam diam dolore dolores duo eirmod eos erat, et nonumy sed tempor et et invidunt justo labore Stet clita ea et gubergren, kasd magna no rebum. sanctus sea sed takimata ut vero voluptua. est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur"
path_local_checkpoints = "PATH/TO/YOUR/LOCAL/CHECKPOINTS"  # Update this path to your local checkpoints directory

def store_local(config, config_name, artifact_type):
    config['device'] = local_config['local_device']
    model_name = config['model_name']
    if artifact_type == "ch_time":
        optim_name, lr, p_art = chosen_model_params_frog[model_name]
    elif artifact_type == "bone":
        optim_name, lr, p_art = chosen_model_params_dog[model_name]
    
    config['ckpt_path'] = f"{path_local_checkpoints}/imagenet_attacked_clean_{artifact_type}_{model_name}_{optim_name}_lr{lr}_p_artifact{p_art}.pth"
    config['batch_size'] = local_config['local_batch_size']
    config['data_paths'] = [local_config['imagenet_dir']]
    config['checkpoint_dir_corrected'] = path_local_checkpoints
    config['dir_precomputed_data'] = local_config['dir_precomputed_data']
    config['only_val'] = True

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def store_cluster(config, config_name, artifact_type):
    model_name = config['model_name']
    config['batch_size'] = 32
    config["device"] = "cuda"

    if artifact_type == "ch_time":
        optim_name, lr, p_art = chosen_model_params_frog[model_name]
    elif artifact_type == "bone":
        optim_name, lr, p_art = chosen_model_params_dog[model_name]

    config['ckpt_path'] = f"checkpoints/imagenet_attacked_clean_{artifact_type}_{model_name}_{optim_name}_lr{lr}_p_artifact{p_art}.pth"
    config['data_paths'] = ["/mnt/imagenet"]
    config['checkpoint_dir_corrected'] = "/mnt/models_corrected"
    config['dir_precomputed_data'] = "/mnt/models_corrected"
    config['only_val'] = False

    with open(f"{config_dir}/cluster/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

for model_name, layer_names in [
    ('vgg16', LAYER_NAMES_BY_MODEL["vgg16"][-1:]),
    ('resnet50d', LAYER_NAMES_BY_MODEL["resnet"][-1:]),
]:
    
    base_config['model_name'] = model_name
    
    p_artifact_rates = [ 0.3] #0.4, 0.5

    for p_artifact in p_artifact_rates:
        for layer_name in layer_names:
            base_config['layer_name'] = layer_name

            ## Baseline Attacked Dog
            config_dog = copy.deepcopy(base_config)
            method = 'Vanilla'
            config_dog["direction_mode"] = "signal"
            config_dog["cav_mode"] = "cavs_max"
            config_dog["cav_scope"] = [0]
            config_dog['method'] = method
            config_dog['lamb'] = 0.0
            config_dog['num_epochs'] = 0
            config_dog['p_artifact'] = p_artifact

            optim_name, lr, backdoor = chosen_model_params_frog[model_name]
            config_dog['optimizer'] = optim_name
            config_dog['lr'] = lr 

            artifact_type_dog = 'bone'
            config_dog['artifact_type'] = artifact_type_dog
            config_dog['attacked_classes'] = ['Dog']


            config_name = f"imagenet_attacked_clean_{artifact_type_dog}_{model_name}_p_artifact{p_artifact}_{layer_name}_baseline"
            store_local(config_dog, config_name, artifact_type_dog)
            store_cluster(config_dog, config_name, artifact_type_dog)

            ## Baseline Attacked Frog
            config_frog = copy.deepcopy(base_config)
            method = 'Vanilla'
            config_frog["direction_mode"] = "signal"
            config_frog["cav_mode"] = "cavs_max"
            config_frog["cav_scope"] = None
            config_frog['method'] = method
            config_frog['lamb'] = 0.0
            config_frog['num_epochs'] = 0
            config_frog['p_artifact'] = p_artifact

            optim_name, lr, backdoor = chosen_model_params_dog[model_name]
            config_frog['optimizer'] = optim_name
            config_frog['lr'] = lr

            artifact_type_frog = 'ch_time'
            config_frog['artifact_type'] = artifact_type_frog
            # config_dog['lsb_trigger'] = lsb_trigger
            # config_dog['lsb_factor'] = 4
            config_frog['time_format'] = "datetime"
            config_frog['attacked_classes'] = ['Frog']

            config_name = f"imagenet_attacked_clean_{artifact_type_frog}_{model_name}_p_artifact{p_artifact}_{layer_name}_baseline"
            #store_local(config_frog, config_name, artifact_type_frog)
            #store_cluster(config_frog, config_name, artifact_type_frog)