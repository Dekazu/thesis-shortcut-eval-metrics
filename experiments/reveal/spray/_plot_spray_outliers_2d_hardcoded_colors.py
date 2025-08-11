from argparse import ArgumentParser
import copy

import numpy as np
from datasets import load_dataset
import h5py
from utils.helper import load_config, none_or_int
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torchvision.utils import make_grid
from sklearn.neighbors import LocalOutlierFactor


SAMPLE_ID_COLOR_MAPS = {
    # ResNet identity 2
    # 1: [1717, 1676, 1771, 1730, 1738, 1706], # ruler
    # 4: [11658, 12762, 14791, 15851, 16105, 18719] # generic micro

    # ResNet input identity
    3: [10698, 10934, 10989, 11264, 11907, 12386,
        202, 10535, 11215, 12215, 12240, 13140
        ]
}

IRRELEVANT_SAMPLE_ID_COLOR_MAPS = {
    1: [1717, 1676, 1771, 1730, 1738, 1706, 1700, 1666, 1780, 1644, 1712, 1412, 1482, 1777, 1767, 1675, 1480, 1727]
}

def get_parser():
    parser = ArgumentParser(
        description='Plot SpRAy 2D embedding with outliers.', )

    parser.add_argument('--n', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--class_id', default=0, type=none_or_int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--savedir', default="plot_files/spray_outliers", type=str)
    parser.add_argument('--analysis_path', default="/Users/rustler/Documents/Projects/bachelor-thesis/results/spray")
    parser.add_argument('--config_file',
                        default="config_files/revealing/isic/local/resnet50d_input_identity.yaml"
                        # default="config_files/revealing/isic/local/resnet50d_identity_2.yaml"
                        # default="config_files/revealing/chexpert/local/vgg16_binaryTarget-Cardiomegaly_pm_features.22.yaml"
                        )
   

    return parser

def main():
    args = get_parser().parse_args()
    config = load_config(args.config_file)
    plot_spray_embedding(config, args.analysis_path, args.split, args.class_id, args.savedir)


def plot_spray_embedding(config, analysis_path, split, class_id, savedir):
    
    
    dataset = load_dataset(config, normalize_data=True)

    splits = {
        "train": dataset.idxs_train,
        "val": dataset.idxs_val,
        "test": dataset.idxs_test,
        }

    dataset = dataset if (split is None) or (split=="all") else dataset.get_subset_by_idxs(splits[split])
    show_ruler_spray_samples = False
    if show_ruler_spray_samples:
        outlier_sample_ids = ",".join([str(i) for i in dataset.sample_ids_by_artifact["ruler_spray_7"]])
    analysis_file = f"{analysis_path}/{config['dataset_name']}/{config['config_name']}_{split}.hdf5"
    
    outlier_samples_original = []
    for k, outliers in SAMPLE_ID_COLOR_MAPS.items():
        outlier_samples_original += outliers

    for emb_type in ["umap", "tsne"]:
    # emb_type = "umap"
        with h5py.File(analysis_file, 'r') as fp:
            sample_ids = np.array(fp[str(class_id)]['index'])
            data_2d = fp[str(class_id)]['embedding'][emb_type][::1]

        sample_id_map = {id_orig: i for i, id_orig in enumerate(sample_ids)}
        sample_id_color_map_run = copy.deepcopy(SAMPLE_ID_COLOR_MAPS)
        irrelevant_sample_id_color_map_run = copy.deepcopy(IRRELEVANT_SAMPLE_ID_COLOR_MAPS)
        # dataset = dataset.get_subset_by_idxs(sample_ids)
        
        outlier_samples = []
        
        for k, outliers in sample_id_color_map_run.items():
            sample_id_color_map_run[k] = [sample_id_map[outlier_id] for outlier_id in outliers if outlier_id in sample_id_map]
            outlier_samples += sample_id_color_map_run[k]

        # noteworthy_sample_ids = []
        for k, samples in irrelevant_sample_id_color_map_run.items():
            irrelevant_sample_id_color_map_run[k] = [sample_id_map[sid] for sid in samples if sid in sample_id_map]
            # outlier_samples += sample_id_color_map_run[k]

        print(f"Potential outliers: {outlier_samples}")    
                
        def get_outlier_label(c):
            for k, outliers in sample_id_color_map_run.items():
                if c in outliers:
                    return k
            return -1
        
        def get_other_label(c):
            for k, sids in irrelevant_sample_id_color_map_run.items():
                if c in sids:
                    return k
            return -1
            
                
        outlier_labels = np.array([get_outlier_label(x) for x in range(0, len(data_2d))])
        other_labels = np.array([get_other_label(x) for x in range(0, len(data_2d))])


        savename = f"{savedir}/{config['dataset_name']}/{config['model_name']}/custom_spray_{config['dataset_name']}_{config['layer_name']}_{emb_type}"
        savename += "" if class_id is None else f"_class{class_id}"
        str_concept_ids = "_".join([str(nid) for nid in outlier_samples_original[:3]])
        savename = f"{savename}_{str_concept_ids}"
        os.makedirs(os.path.dirname(savename), exist_ok=True)


        outlier_imgs = [dataset.reverse_normalization(dataset[i][0]) for i in outlier_samples_original]
        print(f"Store as {savename}")
        create_plot(data_2d, outlier_labels, other_labels, outlier_samples, outlier_imgs, emb_type, f"{savename}.pdf")

def show_outlier_imgs(imgs, ax):
    # IMGS_PER_AX = int(len(imgs) // len(axs))
    # for i, ax in enumerate(axs):
    #     imgs_ax = imgs[i * IMGS_PER_AX:(i+1)*IMGS_PER_AX]
    grid = make_grid([img for img in imgs], padding=2, nrow=3)
    ax.imshow(grid.numpy().transpose(1,2,0))
    ax.set_yticks([]); ax.set_xticks([])

def plot_2d(data, outlier_label, other_label, ax, axis_labels={"x": "UMAP 1", "y": "UMAP 2"}):
    data = pd.DataFrame(data)
    data.columns = ['x', 'y']
    data['outlier_label'] = outlier_label
    data['other_label'] = other_label
    
    # palette =sns.color_palette()[:len(np.unique(data['label'].values))]
    labels = sorted(np.unique(data['outlier_label'].values))[1:]
    palette_outlier = [sns.color_palette()[l] for l in labels]

    labels = sorted(np.unique(data['other_label'].values))[1:]
    palette_others = [sns.color_palette()[l] for l in labels]

    idx_irrelevant = (data["outlier_label"] == -1) & (data["other_label"] == -1)
    idxs_others = (data["outlier_label"] == -1) & (data["other_label"] != -1)
    idxs_relevant = (data["outlier_label"] != -1)

    # irrelevant concepts
    sns.scatterplot(data=data[idx_irrelevant], x="x", y="y", 
                    color="grey", s=10, alpha=.4, ax=ax, legend=False)

    # irrelevant but noteworthy
    sns.scatterplot(data=data[idxs_others].sort_values("other_label"), x="x", y="y", 
                    hue = 'other_label', 
                    palette = palette_others,

                    s=75, alpha=.7, ax=ax, legend=False)


    sns.scatterplot(
        data=data[idxs_relevant].sort_values("outlier_label"), x="x", y="y", 
        hue = 'outlier_label', 
        palette = palette_outlier,
        s=200, alpha = 0.8, legend = False,
        ax = ax
    )

    ax.set(xlabel=axis_labels["x"], ylabel=axis_labels["y"])
    sns.despine()

def create_plot(data, outlier_labels, other_labels, outlier_samples, outlier_imgs, algorithm, savename):
    # nconcepts = 3 
    nrows = 3 #1 + nconcepts + 1
    ncols = 1
    base_size = 1.8
    mul_umap = 5
    gap = .1

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            # figsize=(base_size * mul_umap, (nrows-3) * base_size + base_size * mul_umap), 
                            figsize=(base_size * mul_umap, base_size * mul_umap * 1.5), 
                            # gridspec_kw={'height_ratios':[mul_umap] + [gap] + [1] * nconcepts})
                            gridspec_kw={'height_ratios':[mul_umap] + [gap] + [mul_umap / 2]})

    ax_umap = axs[0]
    plot_2d(data, outlier_labels, other_labels, ax_umap, axis_labels=
            {
                "x": f"{algorithm.upper()} 1",
                "y": f"{algorithm.upper()} 2"
             }
            )
    axs[1].axis("off"); 

    # ref_imgs_reduced = {c: ref_imgs[c] for c in concepts}
    # axs_relmax = axs[2:2+nconcepts]
    show_outlier_imgs(outlier_imgs, axs[2])
    # show_relmax_refimgs(ref_imgs_reduced, axs_relmax)
    # axs_relmax[0].set_title("RelMax")


    fig.savefig(savename, bbox_inches="tight", dpi=300)

    ax_umap.axis("off")

    fig.savefig(savename[:-4] + f"_no_axis.{savename[-3:]}", bbox_inches="tight", dpi=300)
    
if __name__ == "__main__":
    main()
