import copy
import gc
import logging
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision.transforms as T
import tqdm
import wandb
import yaml
from matplotlib import pyplot as plt

from datasets import get_dataset, get_dataset_kwargs
from model_correction import get_correction_method
from models import get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs

from lucent.optvis import objectives
from utils.activation_maximization import render_vis

torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file',
                        default="config_files/clarc/cmnist/local/vgg11_attacked0_color_mnist_input_identity.yaml")
    parser.add_argument('--cav_type', type=str, default=None)
    parser.add_argument('--direction_type', type=str, default=None)
    parser.add_argument('--method', type=str, default=None)
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    print(args.config_file)
    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["wandb_id"] = os.path.basename(args.config_file)[:-5]
            #config_name = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    config['config_file'] = args.config_file

    if args.cav_type:
        cav_mode = args.cav_type
        direction_mode = args.direction_type
        method = args.method
        config['method'] = method
        config["lamb"] = 1
        config["cav_mode"] = cav_mode
        config["direction_mode"] = direction_mode
        #config_name = f"{config_name}_{method}_{cav_mode}_{direction_mode}"


    method = config.get("method", "")
    if (method == "") or ("aclarc" in method.lower()):
        visualize_cav(config)
    else:
        logger.info(f"Skipping TCAV metric for method {method}")


def visualize_cav(config):
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.get("device", default_device)
    dataset_name = config['dataset_name'] + "_attacked" if "attacked" not in config['dataset_name'] and "clevr" not in config['dataset_name']  else config[
            'dataset_name']
    model_name = config['model_name']
    layer_name = config["layer_name"]

    data_paths = config['data_paths']
    img_size = config.get('img_size', 224)
    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)

    ## Local testing
        # config["method"] = "AClarc"
        # config["lamb"] = 1.0
        # config["direction_mode"] = "svm"
        # config["cav_mode"] = "cavs_max"

    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        p_artifact=config.get('p_artifact', 1.0),
                                        image_size=img_size,
                                        artifact_type=config.get('artifact_type', None),
                                        attacked_classes=config.get('attacked_classes', []),
                                        **artifact_kwargs, **dataset_specific_kwargs)
    
    n_classes = len(dataset.classes)
    ckpt_path = config['ckpt_path']
    method = config["method"]
    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path, device=device).to(device)
    kwargs_correction = {}
    artifact_idxs_train = [i for i in dataset.idxs_train if i in dataset.sample_ids_by_artifact[config['artifact']]]
    kwargs_correction['classes'] = dataset.classes
    kwargs_correction['artifact_sample_ids'] = artifact_idxs_train
    kwargs_correction['sample_ids'] = dataset.idxs_train
    kwargs_correction['mode'] = config["cav_mode"]

    correction_method = get_correction_method(method)
    model_corrected = correction_method(model, config, **kwargs_correction)
    cav = model_corrected.cav.clone().detach().cpu().reshape(-1).numpy()

    direction_mode = config['direction_mode']
    savepath = f"results/cav_visualizations/{model_name}_{layer_name}_{direction_mode}.jpg"
    visualize_cav_latent(cav, model, 5, dataset, layer_name, savepath)
    wandb.log({f"CAV visualization ({direction_mode})": wandb.Image(savepath)})

def visualize_cav_latent(cav, model, num_concepts, dataset, layer_name, savepath):
    top_idxs = (-cav).argsort()[:num_concepts]
    channel = lambda n: objectives.channel(layer_name.replace(".", "_"), n)
    images = {}
    for target in top_idxs:
        images[target] = render_vis(model, channel(target), show_image=False, fn_normalize=dataset.normalize_fn)

    obj = sum([channel(t) for t in top_idxs])
    images["all"] = render_vis(model, obj, show_image=False, fn_normalize=dataset.normalize_fn)
    plot_cav_latent(images, savepath)
    
def plot_cav_latent(dict_images, savepath):
    nrows = 1
    ncols = len(dict_images)
    size = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size * ncols, size * nrows), squeeze=True)

    for i, (name, img) in enumerate(dict_images.items()):
        ax = axs[i]
        ax.imshow(img[0].squeeze())
        ax.set_title(name)
        ax.axis("off")

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=300)
    plt.close()

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
