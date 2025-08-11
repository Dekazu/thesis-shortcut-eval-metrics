import copy
import glob
import logging
import os
from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
import torch
import zennit.image as zimage
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from utils.run_lrp import forward_lrp_dl
from zennit.composites import EpsilonPlusFlat

from datasets import do_custom_split, get_dataset
from models import get_canonizer, get_fn_model_loader


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="vgg16")
    parser.add_argument("--ckpt_path")

    parser.add_argument("--dataset_name")
    parser.add_argument("--data_path")
    parser.add_argument("--fname_idx_train", default='data/idxs_train.npy')
    parser.add_argument("--path_samples_mel_attacked", default='data/attacked_mel_samples')
    parser.add_argument("--split", default="train", choices=['train', 'val'])

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--results_dir")
    return parser


def get_attacked_mel_dataset(dataset, path_samples_mel_attacked):
    fnames_samples_ch_mel = [fname.split('/')[-1][:-4] for fname in glob.glob(f"{path_samples_mel_attacked}/*")]
    logger.info(f"Found {len(fnames_samples_ch_mel)} attacked MEL samples in path {path_samples_mel_attacked}.")
    idxs_mel_pois = [i for i, path in enumerate(dataset.ids.image.values) if path in fnames_samples_ch_mel]
    logger.info(f"Found {len(idxs_mel_pois)} according idxs in dataset.")
    dataset_mel_clean = dataset.get_subset_by_idxs(idxs_mel_pois)
    dataset_mel_attacked = copy.deepcopy(dataset_mel_clean)
    dataset_mel_attacked.train_dir = path_samples_mel_attacked
    return dataset_mel_clean, dataset_mel_attacked


def main():
    args = get_parser().parse_args()

    # Read Arguments
    model_name = args.model_name
    ckpt_path = args.ckpt_path
    dataset_name = args.dataset_name
    data_path = args.data_path
    split = args.split
    fname_idx_train = args.fname_idx_train
    path_samples_mel_attacked = args.path_samples_mel_attacked
    device = args.device
    batch_size = args.batch_size
    results_dir = args.results_dir

    run_evaluation(model_name, ckpt_path, dataset_name, data_path, split, fname_idx_train, path_samples_mel_attacked,
                   device, batch_size, results_dir)


def run_evaluation(
        model_name: str,
        ckpt_path: str,
        dataset_name: str,
        data_path: str,
        split: str,
        fname_idx_train: str,
        path_samples_mel_attacked: str,
        device: str,
        batch_size: int,
        results_dir: str
):
    # Load Data
    dataset = get_dataset(dataset_name)(path=data_path, normalize_data=True, split="train")
    dataset_train, dataset_val = do_custom_split(dataset, fname_idx_train)
    dataset_split = dataset_train if split == 'train' else dataset_val

    # Get Clean/Attacked datasets
    dataset_clean, dataset_attacked = get_attacked_mel_dataset(dataset_split, f"{path_samples_mel_attacked}/{split}")

    # Load Model
    fn_model_loader = get_fn_model_loader(model_name)

    model = fn_model_loader(
        ckpt_path=ckpt_path,
        pretrained=False,
        n_class=len(dataset.classes),
        device=device)

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers=canonizers)

    # Run Evaluations
    dl_clean = DataLoader(dataset_clean, batch_size=batch_size, shuffle=False, num_workers=8)
    dl_attacked = DataLoader(dataset_attacked, batch_size=batch_size, shuffle=False, num_workers=8)

    outs_clean, attribs_clean = forward_lrp_dl(model, composite, dl_clean, 'pred', device)
    outs_attacked, attribs_attacked = forward_lrp_dl(model, composite, dl_attacked, 'pred', device)

    # Store Results
    visualize_logits(outs_clean, outs_attacked, dataset_clean.classes, f"{results_dir}/{split}")
    store_heatmaps(dataset_clean, dataset_attacked, attribs_clean, attribs_attacked,
                   outs_clean, outs_attacked, savedir=f"{results_dir}/{split}/attributions")


def normalize_heatmap(heatmap):
    amax = heatmap.max((0, 1), keepdims=True)
    heatmap = (heatmap + amax) / 2 / amax
    return heatmap


def visualize_logits(outs_clean, outs_attacked, classes, savedir):
    size = 4
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(2 * size * 1.5, size))
    data_clean = pd.DataFrame(outs_clean, columns=classes).melt(var_name='label', value_name='logit')
    sns.boxplot(data=data_clean, y='logit', x='label', ax=ax1)
    ax1.set_title("Clean Data")

    data_attacked = pd.DataFrame(outs_attacked, columns=classes).melt(var_name='label', value_name='logit')
    sns.boxplot(data=data_attacked, y='logit', x='label', ax=ax2)
    ax2.set_title("Attacked Data")

    os.makedirs(savedir, exist_ok=True)
    fig.savefig(f"{savedir}/average_logits.png", bbox_inches='tight')
    plt.close()


def store_heatmaps(dataset_clean, dataset_attacked, attribs_clean, attribs_attacked, outs_clean, outs_attacked,
                   level=1.0, cmap='coldnhot', savedir=None):
    denormalizer = dataset_clean.denormalizer
    class_id_to_name_map = dataset_clean.class_id_to_name_map

    for i in range(len(dataset_clean)):
        nrows, ncols = 1, 4
        size = 2
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size))

        input_sample_clean, input_y_clean = dataset_clean[i]
        axs[0].imshow(torch.moveaxis(denormalizer(input_sample_clean), 0, 2))
        axs[0].set_title(f"Label: {class_id_to_name_map[input_y_clean.item()]}")

        img_heatmap_clean = zimage.imgify(normalize_heatmap(attribs_clean[i].sum(0).numpy()), vmin=0., vmax=1.,
                                          level=level, cmap=cmap)
        axs[1].imshow(img_heatmap_clean)
        pred_clean = outs_clean[i].argmax().item()
        axs[1].set_title(f"Pred: {class_id_to_name_map[pred_clean]} ({outs_clean[i][pred_clean]:.1f})")

        input_sample_attacked, input_y_attacked = dataset_attacked[i]
        axs[2].imshow(torch.moveaxis(denormalizer(input_sample_attacked), 0, 2))
        axs[2].set_title(f"Label: {class_id_to_name_map[input_y_attacked.item()]}")

        img_heatmap_attacked = zimage.imgify(normalize_heatmap(attribs_attacked[i].sum(0).numpy()), vmin=0., vmax=1.,
                                             level=level, cmap=cmap)
        axs[3].imshow(img_heatmap_attacked)
        pred_attacked = outs_attacked[i].argmax().item()
        axs[3].set_title(f"Pred: {class_id_to_name_map[pred_attacked]} ({outs_attacked[i][pred_attacked]:.1f})")

        _ = [ax.axis('off') for ax in axs]

        os.makedirs(savedir, exist_ok=True)
        fig.savefig(f"{savedir}/img_{i}.png", bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
