import os
import shutil
from argparse import ArgumentParser
import numpy as np
import torch
from PIL import Image
from crp.attribution import CondAttribution
from matplotlib import pyplot as plt
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat

from datasets import load_dataset
from torch.utils.data import Subset, DataLoader
from models import get_canonizer, get_fn_model_loader
from utils.cav_utils import get_cav_from_model, load_cav
from utils.helper import load_config
from skimage.filters import threshold_otsu

from utils.localization import binarize_heatmaps, get_localizations

MAX_IMGS_SHOW = 6

HARD_CODED_IDS = {
    "ruler": [0,1,2,32,8,33,10],
    "band_aid": [0,2,40,49,50,56],
    "pm": [0,4,17,20,23,25],
    "tube": [1,3,8,18,34,56]
}

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--split", default="all")
    parser.add_argument("--save_dir", default="results/localization")
    parser.add_argument("--direction_mode", default="signal")
    parser.add_argument("--save_localization", default=True, type=bool)
    parser.add_argument("--save_examples", default=True, type=bool)
    parser.add_argument('--cav_type', type=str, default=None)
    parser.add_argument('--config_file', 
                        # default="config_files/revealing/isic/local/resnet50d_identity_2.yaml")
                        # default="config_files/revealing/isic/local/vgg16_features.22.yaml")
                        default="config_files/evaluate_metrics_clean/imagenet_attacked/local/imagenet_attacked_bone_resnet50d_p_artifact0.3_last_conv_baseline.yaml")
                        # default="config_files/revealing/chexpert/local/vgg16_binaryTarget-Cardiomegaly_pm_features.22.yaml")
                        # default="config_files/revealing/hyper_kvasir/local/vgg16_features.29.yaml")
    parser.add_argument("--before_correction", default=False, action="store_true")
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    config = load_config(args.config_file)
    config["direction_mode"] = args.direction_mode
    if config["dataset_name"] == "imagenet_attacked":
        config["batch_size"] = 4

    localize_artifacts(config,
                       split=args.split,
                       save_dir=args.save_dir,
                       save_examples=args.save_examples,
                       save_localization=args.save_localization,
                       before_correction=args.before_correction)

def collate_artifacts(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return images, labels

#
def localize_artifacts(config: dict,
                       split: str,
                       save_dir: str,
                       save_examples: bool,
                       save_localization: bool,
                       before_correction: bool = False):
    """Spatially localize artifacts in input samples.

    Args:
        config (dict): experiment config
        split (str): data split to use
        artifact (str): artifact
        save_dir (str): save_dir
        save_examples (bool): Store example images
        save_localization (bool): Store localization heatmaps
    """

    dataset_name = config['dataset_name']
    model_name = config['model_name']
    artifact = config.get("artifact_type", None)
    artifact_type = config['artifact_type']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.get("device", device)
    
    cav_mode = config.get("cav_mode", "cavs_max")
    direction_mode = config['direction_mode']
    

    dataset = load_dataset(config, normalize_data=True)

    assert artifact in dataset.sample_ids_by_artifact.keys(), f"Artifact {artifact} unknown."

    artifact_ids = dataset.sample_ids_by_artifact[artifact]

    print(f"Chose {len(artifact_ids)} target samples.")


    ckpt_path = config['ckpt_path'] if before_correction else f"{config['checkpoint_dir_corrected']}/{config['config_name']}/last.ckpt"
    model = get_fn_model_loader(model_name=model_name)(n_class=len(dataset.classes),
                                                       ckpt_path=ckpt_path, device=device)
    model = model.to(device)
    model.eval()

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)
    attribution = CondAttribution(model)

    img_to_plt = lambda x: dataset.reverse_normalization(x.detach().cpu()).permute((1, 2, 0)).int().numpy()
    hm_to_plt = lambda x: x.detach().cpu().numpy()

    if config["model_name"] == "vgg16":
        layer_names = ["features.27", "features.28"]
    elif config["model_name"] == "resnet50d":
        layer_names = ["identity_2"]
    else:
        raise ValueError(f"Unknown model: {config['model_name']}")


    keys = ["standard", "perfect_cav"]
    if config["dataset_name"] == "isic_attacked" or config["dataset_name"] == "waterbirds":
        keys.append("cav_scope")

    for layer_name, key in ((ln, k) for ln in layer_names for k in keys):
        config["layer_name"] = layer_name
        suffix = f"_{key}"

        ## get CAV
        config_for_cav = config.copy()
        if "perfect_cav" in key:
            config_for_cav["cav_scope"] = None
            if config["dataset_name"] == "imagenet_attacked" or config["dataset_name"] == "isic_attacked":
                config_for_cav["dataset_name"] = f"perfect_cav_{config['dataset_name']}"
            elif config["dataset_name"] == "waterbirds":
                config_for_cav["dataset_name"] = "backgrounds"
            w = load_cav(config_for_cav, artifact, mode=cav_mode)
        else:
            if "cav_scope" in key:
                config_for_cav["cav_scope"] = [1]
            w = get_cav_from_model(model, dataset, config_for_cav, artifact, store_cav=True)

        # ── Efficient DataLoader over just artifact samples
        subset = Subset(dataset, artifact_ids)
        loader = DataLoader(
            subset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_artifacts
        )

        # for saving the first few examples
        saved_examples = 0
        example_imgs, example_hms, example_hmb = [], [], []

        # ── Batch loop ──
        for batch_idx, (data, _) in enumerate(tqdm(loader, desc=f"Layer {layer_name}")):
            data = data.to(device, non_blocking=True)

            # compute attributions (grad enabled, so no accidental grad accumulation)
            with torch.enable_grad():
                attr, _ = get_localizations(data, w, attribution, composite, config, device)

            # move heatmaps to CPU+numpy immediately
            hms = [h.detach().cpu().numpy()             for h in attr.heatmap]
            hms_clamped = [np.clip(h, 0, None)         for h in hms]

            # ── Per-sample saving & example gathering ──
            for i, hm in enumerate(hms_clamped):
                # map back to original dataset sample ID
                global_idx = loader.dataset.indices[batch_idx * config["batch_size"] + i]
                sample_id = int(global_idx)

                if save_localization:
                    # original heatmap → 8-bit
                    hm8 = (hm / (hm.max() + 1e-8) * 255).astype(np.uint8)
                    out_dir  = os.path.join(save_dir, "heatmaps",    dataset_name, model_name,
                                            f"{artifact_type}_{layer_name}_{config['direction_mode']}{suffix}")
                    out_dir_b= os.path.join(save_dir, "heatmaps_binary", dataset_name, model_name,
                                            f"{artifact_type}_{layer_name}_{config['direction_mode']}{suffix}")
                    os.makedirs(out_dir,  exist_ok=True)
                    os.makedirs(out_dir_b, exist_ok=True)

                    # save grayscale heatmap
                    Image.fromarray(hm8).save(os.path.join(out_dir,  f"{sample_id}.png"))

                    # Otsu binarization & save
                    th = threshold_otsu(hm8)
                    bin_mask = ((hm8 > th).astype(np.uint8) * 255)
                    Image.fromarray(bin_mask).save(os.path.join(out_dir_b, f"{sample_id}.png"))

                # stash a few for example plots
                if save_examples and saved_examples < MAX_IMGS_SHOW:
                    img = dataset.reverse_normalization(data[i].cpu()).permute(1,2,0).int().numpy()
                    example_imgs.append(img)
                    example_hms.append(hms[i])
                    example_hmb.append(bin_mask if save_localization else (hms_clamped[i] > th).astype(np.uint8))
                    saved_examples += 1

            # cleanup to free GPU memory
            del attr, hms, hms_clamped, data
            torch.cuda.empty_cache()

        # ── Plot the few saved examples ──
        if save_examples and example_imgs:
            mname_map = {"resnet50d":"ResNet-50","vgg16":"VGG-16"}
            display_name = mname_map.get(model_name, model_name)
            pdf_path = os.path.join(
                save_dir, "examples_all", dataset_name, model_name,
                f"{artifact_type}_{layer_name}_{split}{suffix}.pdf"
            )
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            plot_localization(
                example_imgs,
                example_hms,
                example_hmb,
                pdf_path,
                display_name,
                max_imgs=len(example_imgs)
            )
            svg_path = os.path.join(
                save_dir, "examples_all", dataset_name, model_name,
                f"{artifact_type}_{layer_name}_{split}{suffix}.svg"
            )
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            plot_localization(
                example_imgs,
                example_hms,
                example_hmb,
                svg_path,
                display_name,
                max_imgs=len(example_imgs)
            )

def plot_localization(x, hm, loc_pred, savename, model_name, max_imgs=6):
    import matplotlib.pyplot as plt

    ncols = min(max_imgs, len(x))
    fig, axs = plt.subplots(3, ncols, figsize=(ncols*1.7, 3*1.7))
    for i in range(ncols):
        # input image
        ax = axs[0, i]
        ax.imshow(x[i]); ax.set_xticks([]); ax.set_yticks([])
        if i==0: axs[0,0].set_ylabel("Input")

        # heatmap
        ax = axs[1, i]
        v = np.abs(hm[i]).max()
        ax.imshow(hm[i], cmap="bwr", vmin=-v, vmax=v)
        ax.set_xticks([]); ax.set_yticks([])
        if i==0: axs[1,0].set_ylabel("Heatmap")

        # binary mask
        ax = axs[2, i]
        ax.imshow(loc_pred[i], cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        if i==0: axs[2,0].set_ylabel("Mask")

    fig.suptitle(model_name, fontsize=16, y=0.95)
    plt.tight_layout()
    fig.savefig(savename, bbox_inches="tight", dpi=400)
    plt.close(fig)

if __name__ == "__main__":
    main()
