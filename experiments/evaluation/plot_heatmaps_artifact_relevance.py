from argparse import ArgumentParser
from utils.helper import load_config
import os
import wandb
import torch
from datasets import load_dataset
from models import get_fn_model_loader
from crp.attribution import CondAttribution
from models import get_canonizer
from zennit.composites import EpsilonPlusFlat
import matplotlib.pyplot as plt
import torchvision
from crp.helper import get_layer_names
import numpy as np
import random


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--sample_ids", default=None, type=str)
    parser.add_argument("--normalized", default="max", type=str)
    parser.add_argument("--results_dir", default="results/plot_artifact_relevance", type=str)
    parser.add_argument('--config_file',
                        default="config_files/evaluate_metrics_clean/isic_binary/local/isic_binary_vgg16_p_artifact0.5_p_artifact_na0.05_features.29_baseline.yaml"
                        )
    parser.add_argument('--before_correction', action="store_true")
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    config = load_config(args.config_file)

    plot_to_wandb = False
    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=f"{config['wandb_project_name']}", resume=True)
        plot_to_wandb = True

    sample_ids = [int(i) for i in args.sample_ids.split(",")] if args.sample_ids else None

    plot_heatmap_artifact_relevance(config, args.before_correction, sample_ids, args.normalized, plot_to_wandb, args.results_dir)

def plot_heatmap_artifact_relevance(config, before_correction, sample_ids, normalized, plot_to_wandb, path):
    dataset_name = config['dataset_name']
    config_name = config['config_name']
    device = "cpu" # TODO: Add Cuda again, but it is not working right now
    dataset = load_dataset(config, normalize_data=True, hm=True)
    artifact_name = config['artifact_type'] if dataset_name == "waterbirds" else config['artifact'] 
    attacked_classes = config.get("attacked_classes", None) 
    if dataset_name == "waterbirds":
        attacked_classes = "waterbirds"
    path = f"{path}/{dataset_name}/{config_name}"

    if sample_ids is None:
        # get 5 random samples
        random.seed(42)
        # if sample_ids is np.array put it to list
        if isinstance(dataset.sample_ids_by_artifact[artifact_name], np.ndarray):
            dataset.sample_ids_by_artifact[artifact_name] = dataset.sample_ids_by_artifact[artifact_name].tolist()
        # get 5 random samples
        if attacked_classes is not None:
            ids_artifact = dataset.sample_ids_by_artifact[artifact_name]
            # Filter out non-attacked classes
            ids_artifact = [i for i in ids_artifact if dataset.classes[dataset[i][1]] in attacked_classes]
            sample_ids = random.sample(ids_artifact, 5)
        else:
            sample_ids = random.sample(dataset.sample_ids_by_artifact[artifact_name], 5)
    
    data = torch.stack([dataset[j][0] for j in sample_ids], dim=0).to(device)
    target = torch.stack([dataset[j][1] for j in sample_ids], dim=0).to(device)
    masks = torch.stack([dataset[j][2] for j in sample_ids], dim=0).to(device)

    ckpt_path = config['ckpt_path'] if before_correction else f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt"

    model = get_fn_model_loader(model_name=config['model_name'])(n_class=dataset.classes.__len__(), ckpt_path=ckpt_path, device=device)
    model.eval()
    
    attribution = CondAttribution(model)
    canonizer = get_canonizer(config['model_name'])
    composite = EpsilonPlusFlat(canonizer)

    condition = [{"y": c_id.item()} for c_id in target]
    attr_model = attribution(data.requires_grad_(), condition, composite)


    max = get_normalization_constant(attr_model, normalized)
    heatmaps = attr_model.heatmap / max
    heatmaps = heatmaps.detach().cpu().numpy()
    mask = masks.detach().cpu().numpy()

    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.ReLU])
    art_rel = artifact_relevance_sample(data, target, masks, composite, attribution, layer_names) 

    # Plot Image + Heatmap + Mask + Artifact Relevance Value
    fig, axs = plt.subplots(len(data), 4, figsize=(20, 5 * len(data)))
    for i, idx in enumerate(sample_ids):
        axs[i, 0].imshow(dataset.reverse_normalization(dataset[idx][0]).permute(1, 2, 0) / 255)
        axs[i, 1].imshow(heatmaps[i], cmap="bwr", vmin=-1, vmax=1)
        axs[i, 2].imshow(mask[i], cmap="gray")
        axs[i, 3].imshow(heatmaps[i] * mask[i], cmap="bwr", vmin=-1, vmax=1)
        axs[i, 0].set_title(f"Sample {sample_ids[i]} -- Target: {dataset.map_target_label(target[i].item())}")
        axs[i, 1].set_title("Heatmap")
        axs[i, 2].set_title("Mask")
        axs[i, 3].set_title(f"Heatmap * Mask: Artifact Relevance is {art_rel[i]:.2f}")
        for j in range(4):
            axs[i, j].axis("off")
        
        # make border thicker
        for ax in axs[i, :]:
            for spine in ax.spines.values():
                spine.set_linewidth(5)
        
    #plt.tight_layout()
    
    if not os.path.exists(path):
        os.makedirs(path)

    # Save as jpeg, png, pdf
    plt.savefig(f"{path}/artifact_relevance_{dataset_name}_{config_name}.png", dpi=400)
    plt.savefig(f"{path}/artifact_relevance_{dataset_name}_{config_name}.pdf", dpi=400)
    plt.savefig(f"{path}/artifact_relevance_{dataset_name}_{config_name}.jpeg", dpi=10)
    plt.savefig(f"{path}/artifact_relevance_{dataset_name}_{config_name}.svg", dpi=400)

    if plot_to_wandb:
        wandb.log({f"artifact_relevance": wandb.Image(f"{path}/artifact_relevance_{dataset_name}_{config_name}.jpeg")})
    plt.show()



def artifact_relevance_sample(data, targets, mask, composite, attribution, layer_names):
    condition = [{"y": c_id} for c_id in targets]
    attr = attribution(data, condition, composite, record_layer=layer_names, init_rel=1)
    gaussian = torchvision.transforms.GaussianBlur(kernel_size=41, sigma=8.0)

    mask = 1.0 * (mask / mask.abs().flatten(start_dim=1).max(1)[0][:, None, None] > 0.1)
    mask = gaussian(mask.clamp(min=0)) ** 1.0
    mask = 1.0 * (mask / mask.abs().flatten(start_dim=1).max(1)[0][:, None, None] > 0.3)

    inside = (attr.heatmap * mask).abs().sum((1, 2)) / (
            attr.heatmap.abs().sum((1, 2)) + 1e-10)
    return list(inside.detach().cpu())

def get_normalization_constant(attr, normalization_mode):
    if normalization_mode == 'max':
        return attr.heatmap.flatten(start_dim=1).max(1, keepdim=True).values[:, None]
    elif normalization_mode == 'abs_max':
        return attr.heatmap.flatten(start_dim=1).abs().max(1, keepdim=True).values[:, None]
    else:
        raise ValueError("Unknown normalization")

if __name__ == '__main__':
    main()