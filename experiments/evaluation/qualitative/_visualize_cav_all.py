import copy
import logging
import os
from argparse import ArgumentParser

import torch
import yaml
from matplotlib import pyplot as plt

from datasets import get_dataset, get_dataset_kwargs
from model_correction import get_correction_method
from models import get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs

from lucent.optvis import objectives
from utils.activation_maximization import render_vis
from utils.layer_names import LAYER_NAMES_BY_MODEL

torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file',
                        default="config_files/clarc/isic_attacked/local/resnet18_lsb_Vanilla_sgd_lr0.0005_input_identity.yaml")
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    print(args.config_file)
    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    config['config_file'] = args.config_file
    visualize_cavs_all(config)


def visualize_cavs_all(config):

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

    config["method"] = "AClarc"
    config["lamb"] = 1.0
    config["cav_mode"] = "cavs_max"

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

    all_cav_visualizations = {}
    direction_modes = [
        "signal", 
        "svm"
        ]
    layer_names_all = LAYER_NAMES_BY_MODEL[model_name]
    NUM_CONCEPTS = 3

    for direction_mode in direction_modes:
        config["direction_mode"] = direction_mode
        all_cavs_direction = dict()

        for layer_name in layer_names_all:
            config["layer_name"] = layer_name
            correction_method = get_correction_method(method)
            model_iter = copy.deepcopy(model)
            model_corrected = correction_method(model_iter, config, **kwargs_correction)
            cav = model_corrected.cav.clone().detach().cpu().reshape(-1).numpy()
            
            all_cavs_direction[layer_name] =  compute_latent_cav_visualization(cav, model_iter, NUM_CONCEPTS, dataset, layer_name)
        all_cav_visualizations[direction_mode] = all_cavs_direction
    savepath = f"results/cav_visualizations/{dataset_name}_{model_name}.png"
    plot_cavs_latent(all_cav_visualizations, NUM_CONCEPTS, savepath)

def compute_latent_cav_visualization(cav, model, num_concepts, dataset, layer_name):
    top_idxs = (-cav).argsort()[:num_concepts]
    channel = lambda n: objectives.channel(layer_name.replace(".", "_"), n)
    images = {}
    for target in top_idxs:
        images[target] = render_vis(model, channel(target), show_image=False, fn_normalize=dataset.normalize_fn)
        # np.random.rand(3, 128, 128)#
    obj = sum([channel(t) for t in top_idxs])
    images["all"] = render_vis(model, obj, show_image=False, fn_normalize=dataset.normalize_fn)
    return images
    
def plot_cavs_latent(all_cav_visualizations, num_concepts, savepath):
    nrows = len(all_cav_visualizations[list(all_cav_visualizations.keys())[0]])
    ncols = len(all_cav_visualizations) * (num_concepts + 1)
    size = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size * ncols, size * nrows), squeeze=False)

    for i_dir, (direction_mode, imgs_layer_dict) in enumerate(all_cav_visualizations.items()):
        for i_layer, (layer_name, imgs_dict) in enumerate(imgs_layer_dict.items()):
            for i_img, (name, img) in enumerate(imgs_dict.items()):
                ax = axs[i_layer][i_dir * (num_concepts+1) + i_img]
                ax.imshow(img[0].squeeze())
                ax.set_title(f"{direction_mode} - {name}")
                if i_img == 0:
                    ax.set_ylabel(layer_name)
                ax.set_xticks([])
                ax.set_yticks([])

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
