#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot two 3×4 TCAV figures—(A) samples whose true label is the attacked class AND
predicted attacked, and (B) samples whose true label is NOT the attacked class BUT
still predicted attacked—using one shared colour‐scale computed from the data.

All sampling is deterministic (seed+sort). Each figure shows (IMG · HEAT · IMG · HEAT)
×3 rows, and pops up the figure on screen *and* saves to disk.
"""

import os
import random
import logging

import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch.utils.data import DataLoader

import wandb
from datasets import load_dataset
from models import get_fn_model_loader
from utils.cav_utils import get_cav_from_model, load_cav
from utils.helper import load_config
import torch.nn.functional as F

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
NUM_SAMPLES = 6    # six images per figure
ROWS, COLS  = 3, 4 # layout: IMG · HEAT · IMG · HEAT  × 3 rows

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ─── ARGPARSE ────────────────────────────────────────────────────────────────
def get_args():
    p = ArgumentParser()
    p.add_argument("--config_file",
                   default="config_files/evaluate_metrics_clean/imagenet_attacked/local/imagenet_attacked_bone_resnet50d_p_artifact0.3_last_conv_baseline.yaml")
    p.add_argument("--before_correction", action="store_true")
    p.add_argument("--results_dir", default="results/tcav_maps")
    return p.parse_args()

# ─── UTILITIES ────────────────────────────────────────────────────────────────
def to_display_img(t: torch.Tensor, ds) -> np.ndarray:
    if hasattr(ds, "reverse_normalization"):
        t = ds.reverse_normalization(t)
    # from (C,H,W) in [0..255] to (H,W,C) in [0..1]
    return t.permute(1, 2, 0).cpu().numpy() / 255.0

def forward_hook(_, __, out):
    global activations
    activations = out
    return out.clone()

def compute_tcav_maps(ds, model, cav, attacked_cls: int,
                      sample_ids: list, device: str):
    subset = ds.get_subset_by_idxs(sample_ids)
    dl = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    maps, means, y_trues, y_preds = [], [], [], []
    for x, y in tqdm.tqdm(dl, desc="TCAV"):
        x = x.to(device)
        x.requires_grad = True

        logits = model(x)
        score = logits[:, attacked_cls]
        grad = torch.autograd.grad(
            outputs=score,
            inputs=activations,
            grad_outputs=torch.ones_like(score),
            retain_graph=False
        )[0].detach().cpu()  # (1, C, Hf, Wf)

        # elementwise * with cav channels, then sum over channels → (1,Hf,Wf)
        tcav_map = (grad * cav[..., None, None]).sum(1)
        maps.append(tcav_map)
        means.append(tcav_map.mean().item())

        model.zero_grad()
        y_trues.append(y.item())
        if logits.shape[1] > 1:
            y_preds.append(int(logits.argmax(dim=1).item()))
        else:
            y_preds.append(int(score.item() > 0))

    return maps, means, y_trues, y_preds

# ─── PLOTTING ────────────────────────────────────────────────────────────────
def plot_tcav_run(ds, sample_ids: list, tcav_maps: list, cfg: dict,
                  run_name: str, results_dir: str, vmin: float,
                  vmax: float, suffix):
    fig, axes = plt.subplots(ROWS, COLS, figsize=(12, 9))
    fig.suptitle(f"TCAV ({run_name}) {suffix}", fontsize=10)
    im = None

    for idx, (img_lbl, tcav) in enumerate(zip(
            ds.get_subset_by_idxs(sample_ids), tcav_maps)):
        img, lbl = img_lbl
        row = idx // 2
        c_img = (idx % 2) * 2
        c_map = c_img + 1

        img_np = to_display_img(img, ds)
        H_img, W_img = img.shape[1:]

        # define a rigid pixel→data mapping
        extent = (0, W_img, H_img, 0)  # (x0, x1, y0, y1) with origin='upper'

        # ── raw image ────────────────────────────────────────────────────
        ax_i = axes[row, c_img]
        ax_i.imshow(
            img_np,
            extent=extent,
            origin='upper',
            interpolation='nearest'
        )
        ax_i.set_title(f"ID {sample_ids[idx]} – {ds.classes[lbl]}", fontsize=9)
        ax_i.axis("off")

        # ── upsample TCAV map with exact corner alignment ─────────────────
        tcav_tensor = tcav.unsqueeze(0)  # shape (1,1,Hf,Wf)
        # tcav_tensor = F.interpolate(
        #     tcav_tensor,
        #     size=(H_img, W_img),
        #     mode='bilinear',
        #     align_corners=True
        # )
        tcav_tensor = tcav_tensor.squeeze().cpu().numpy()  # shape (Hf,Wf)

        # ── overlay heatmap ──────────────────────────────────────────────
        ax_m = axes[row, c_map]
        ax_m.imshow(
            img_np,
            extent=extent,
            origin='upper',
            interpolation='nearest'
        )
        im = ax_m.imshow(
            tcav_tensor,
            cmap="bwr",
            vmin=vmin, vmax=vmax,
            alpha=1,
            extent=extent,
            origin='upper',
            # interpolation='bilinear'
        )
        ax_m.set_aspect('equal')
        ax_m.set_title(f"TCAV map; mean = {1000*tcav.mean():.2f}", fontsize=9)
        ax_m.axis("off")

    # remove any unused axes
    for ax in axes.flatten():
        if not ax.has_data():
            fig.delaxes(ax)

    if im is None:
        raise RuntimeError("No TCAV maps were drawn; cannot add colorbar.")

    # colorbar on the right
    cax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax)

    plt.tight_layout(rect=[0, 0.03, 0.92, 0.97])

    # save outputs
    out_dir = os.path.join(results_dir, cfg["config_name"])
    os.makedirs(out_dir, exist_ok=True)
    fname = f"tcav_{cfg['dataset_name']}_{cfg['artifact_type']}_{run_name}{suffix}"
    for ext, dpi in (("png", 300), ("jpeg", 100), ("pdf", None), ("svg", None)):
        path = os.path.join(out_dir, f"{fname}.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

    # optional W&B logging
    if cfg.get("wandb_api_key"):
        wandb.log({f"{fname}": wandb.Image(fig)})

    plt.show()
    plt.close(fig)


# ─── MAIN SCRIPT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO
    )
    args = get_args()
    cfg  = load_config(args.config_file)

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    cfg["device"] = device

    ds = load_dataset(cfg, normalize_data=True)
    n_cls = len(ds.classes)

    ckpt_path = (cfg["ckpt_path"] if args.before_correction
                 else f"{cfg['checkpoint_dir_corrected']}/"
                      f"{cfg['config_name']}/last.ckpt")

    model = get_fn_model_loader(cfg["model_name"])(
        n_class=n_cls,
        ckpt_path=ckpt_path,
        device=device
    ).to(device)
    model.eval()

    if cfg.get("wandb_api_key"):
        os.environ["WANDB_API_KEY"] = cfg["wandb_api_key"]
        wandb.init(
            id=cfg["wandb_id"],
            project=cfg["wandb_project_name"],
            resume=True
        )

    # pick which layer(s) to hook
    if "resnet" in cfg["model_name"]:
        layer_names = ["identity_2"]
    elif "vgg" in cfg["model_name"]:
        layer_names = ["features.27", "features.28"]
    else:
        raise ValueError(f"Unknown model: {cfg['model_name']}")
    
    keys = ["standard", "perfect_cav"]
    if cfg["dataset_name"] == "isic_attacked" or cfg["dataset_name"] == "waterbirds":
        keys.append("cav_scope")

    for layer_name, key in ((ln, k) for ln in layer_names for k in keys):
        cfg["layer_name"] = layer_name
        suffix = f"_{key}"
        
        config_for_cav = cfg.copy()
        if "perfect_cav" in key:
            config_for_cav["cav_scope"] = None
            if cfg["dataset_name"] == "imagenet_attacked" or cfg["dataset_name"] == "isic_attacked":
                config_for_cav["dataset_name"] = f"perfect_cav_{cfg['dataset_name']}"
            elif cfg["dataset_name"] == "waterbirds":
                config_for_cav["dataset_name"] = "backgrounds"
            cav = load_cav(config_for_cav, cfg["artifact_type"], mode="cavs_max")
        else:
            if "cav_scope" in key:
                config_for_cav["cav_scope"] = [1]

            cav = get_cav_from_model(
                model, ds, cfg, cfg["artifact_type"], mode="cavs_max"
            )

        # register hook
        hook_registered = False
        for name, module in model.named_modules():
            if name.endswith(layer_name):
                module.register_forward_hook(forward_hook)
                hook_registered = True
                break
        if not hook_registered:
            raise RuntimeError(f"Could not find layer {layer_name} in model.")

        # determine attacked class
        if cfg["dataset_name"] == "waterbirds":
            attacked_class = ds.get_class_id_by_name("waterbird")
        else:
            attacked_class = ds.get_class_id_by_name(
                ds.attacked_classes[0]
            )

        artifact_ids = [
            i for i in ds.sample_ids_by_artifact[cfg["artifact_type"]]
            if i in ds.idxs_test
        ]

        tcav_maps, means, y_trues, y_preds = compute_tcav_maps(
            ds, model, cav, attacked_class, artifact_ids, device
        )
        id_to_pos = {idx: i for i, idx in enumerate(artifact_ids)}

        # pick top/bottom and attacked/non-attacked
        sorted_idxs = np.argsort(means)
        top_ids    = [artifact_ids[i] for i in sorted_idxs[-NUM_SAMPLES:]]
        bottom_ids = [artifact_ids[i] for i in sorted_idxs[:NUM_SAMPLES]]

        attacked_ids = [
            idx for idx in artifact_ids
            if y_trues[id_to_pos[idx]] == attacked_class
            and y_preds[id_to_pos[idx]] == attacked_class
        ]
        non_attacked_ids = [
            idx for idx in artifact_ids
            if y_trues[id_to_pos[idx]] != attacked_class
            and y_preds[id_to_pos[idx]] == attacked_class
        ]

        # pad if too few
        if len(attacked_ids) < NUM_SAMPLES:
            pad = [i for i in artifact_ids if i not in attacked_ids]
            attacked_ids += random.sample(pad, NUM_SAMPLES - len(attacked_ids))
        if len(non_attacked_ids) < NUM_SAMPLES:
            pad = [i for i in artifact_ids if i not in non_attacked_ids]
            non_attacked_ids += random.sample(pad, NUM_SAMPLES - len(non_attacked_ids))

        attacked_ids     = attacked_ids[:NUM_SAMPLES]
        non_attacked_ids = non_attacked_ids[:NUM_SAMPLES]

        # global vmin/vmax
        concat = torch.cat([m.flatten() for m in tcav_maps], dim=0)
        abs_max = max(abs(concat.min().item()), abs(concat.max().item()))
        vmin, vmax = -abs_max, abs_max

        # Overwrite vmin/vmax:
        if cfg["dataset_name"] == "waterbirds":
            
            vmin, vmax = -0.08, 0.08
            if key == "perfect_cav":
                vmin, vmax = -0.08, 0.08
                
        elif cfg["dataset_name"] == "isic_attacked":
            vmin, vmax = -0.22, 0.22
        elif cfg["dataset_name"] == "imagenet_attacked":
            vmin, vmax = -0.05, 0.05 # 

        logging.info(f"TCAV vmin={vmin:.5f}, vmax={vmax:.5f}")
        if cfg.get("wandb_api_key"):
            wandb.log({
                f"tcav_vmin_{cfg['dataset_name']}_{cfg['artifact_type']}{suffix}": vmin,
                f"tcav_vmax_{cfg['dataset_name']}_{cfg['artifact_type']}{suffix}": vmax
            })

        run_name = (f"{cfg['model_name']}_{cfg['dataset_name']}_"
                    f"{cfg['artifact_type']}_{layer_name}{suffix}")

        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)

        # plot all four variants
        plot_tcav_run(
            ds, top_ids,
            [tcav_maps[id_to_pos[i]] for i in top_ids],
            cfg, run_name, f"{results_dir}/top/{key}", vmin, vmax,
            suffix=f"_top{suffix}"
        )
        plot_tcav_run(
            ds, bottom_ids,
            [tcav_maps[id_to_pos[i]] for i in bottom_ids],
            cfg, run_name, f"{results_dir}/bottom/{key}", vmin, vmax,
            suffix=f"_bottom{suffix}"
        )
        plot_tcav_run(
            ds, attacked_ids,
            [tcav_maps[id_to_pos[i]] for i in attacked_ids],
            cfg, run_name, f"{results_dir}/attacked/{key}", vmin, vmax,
            suffix=f"_attacked{suffix}"
        )
        plot_tcav_run(
            ds, non_attacked_ids,
            [tcav_maps[id_to_pos[i]] for i in non_attacked_ids],
            cfg, run_name, f"{results_dir}/non_attacked/{key}", vmin, vmax,
            suffix=f"_non_attacked{suffix}"
        )

        logging.info(f"Finished plotting TCAV maps for {run_name}{suffix}.")

        # save all maps (as image) with sample IDs in the filename
        out_dir = os.path.join(results_dir, "all_maps", key)
        os.makedirs(out_dir, exist_ok=True)

        for idx, (sample_id, tcav_map) in enumerate(zip(
                artifact_ids, tcav_maps)):
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(tcav_map.cpu().squeeze(), cmap="bwr", vmin=vmin, vmax=vmax)
            ax.axis("off")

            fname = f"tcav_map_id{sample_id}{suffix}_no_mean.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")

            ax.set_title(f"ID {sample_id} - mean={1000*means[idx]:.2f}")            

            fname = f"tcav_map_id{sample_id}{suffix}.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")

        
