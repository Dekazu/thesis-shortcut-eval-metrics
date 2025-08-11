from argparse import ArgumentParser

import numpy as np
from datasets import DATASET_CLASSES, DATASET_NORM_PARAMS, load_dataset
from models import get_fn_model_loader, get_canonizer
from utils.dimensionality_reduction import get_2d_data
from utils.dora.dora import EA_distance, SignalDataset, SignalDatasetRefData
from utils.dora.model import get_dim, modify_model

from utils.helper import load_config, none_or_int
import os
import torch
import torchvision.transforms as transforms
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from crp.image import zimage
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.visualization import FeatureVisualization
from torchvision.utils import make_grid
import torchvision.transforms as T
from PIL import Image
from sklearn.neighbors import LocalOutlierFactor
from zennit.composites import EpsilonPlusFlat

from utils.render import vis_opaque_img_border

CONCEPT_ID_COLOR_MAPS = {
    # ResNet identity 2
    1: [420,428], # ruler
    2: [346,583], # (white) hair
    # 0: [130,690]

}

IRRELEVANT_CONCEPT_ID_COLOR_MAPS = {
    # ResNet identity 2
    0: [130,690]

}

def get_parser():
    parser = ArgumentParser(
        description='Generate sAMS for DORA Analysis.', )

    parser.add_argument('--n', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--class_id', default=0, type=none_or_int)
    parser.add_argument('--aggr', default="avg", type=str)
    parser.add_argument('--ref_split', default="train", type=str)
    parser.add_argument('--ref_type', default="real_rel", type=str)
    parser.add_argument('--savedir', default="plot_files/dora", type=str)
    parser.add_argument('--config_file',
                        default="config_files/revealing/isic/local/resnet50d_identity_2.yaml")
   

    return parser

def main():
    args = get_parser().parse_args()
    config = load_config(args.config_file)
    plot_dora_with_concepts(config, args.n, args.ref_type, args.aggr, args.batch_size, args.ref_split, args.class_id, args.savedir)


def plot_dora_with_concepts(config, n, ref_type, aggr, batch_size, ref_split, class_id, savedir):
    sams_dir = f"{config['dir_precomputed_data']}/dora_data/{config['dataset_name']}_{config['model_name']}_{aggr}/sAMS/{config['config_name']}/"

    

    model_name = config["model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(DATASET_CLASSES[config["dataset_name"]].classes)
    model = get_fn_model_loader(model_name)(n_class=n_classes,
                                            ckpt_path=config["ckpt_path"]
                                            ).to(device).eval()

    model = modify_model(model, config["layer_name"], aggr="avg")
    k = get_dim(model, config["img_size"], device)

    ## CRP stuff
    model = get_fn_model_loader(model_name)(n_class=n_classes,
                                            ckpt_path=config["ckpt_path"]
                                            ).to(device).eval()
    
    max_mode = "relevance"
    ref_dataset = load_dataset(config, normalize_data=False)

    splits = {
        "train": ref_dataset.idxs_train,
        "val": ref_dataset.idxs_val,
        "test": ref_dataset.idxs_test,
        }

    dataset_name = config['dataset_name']
    ref_split = "test" if dataset_name == "imagenet" else ref_split
    ref_dataset = ref_dataset if (ref_split is None) or (ref_split=="all") else ref_dataset.get_subset_by_idxs(splits[ref_split])
    layer_name = config["layer_name"]
    layer_names = [layer_name]

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()
    layer_map = {layer: cc for layer in layer_names}
    print(f"using layer {layer_name}")
    attribution = CondAttribution(model)
    
    fv_name = f"{config['dir_precomputed_data']}/crp_files/{dataset_name}_{ref_split}_{model_name}"
    if class_id is not None:
        fv_name = f"{fv_name}_class{class_id}"
        idxs_class = [i for i in range(len(ref_dataset)) if ref_dataset.get_target(i) == class_id]
        ref_dataset = ref_dataset.get_subset_by_idxs(idxs_class)

    fv = FeatureVisualization(attribution, ref_dataset, layer_map, preprocess_fn=ref_dataset.normalize_fn,
                            path=fv_name, cache=None)
    
    ref_imgs = fv.get_max_reference([0], layer_name, max_mode, (0, n), composite=composite, rf=True,
                                plot_fn=vis_opaque_img_border)
    
    mean, std = DATASET_NORM_PARAMS[config['dataset_name']]
    fn_normalize = transforms.Normalize(mean=mean,
                                        std=std)
    if ref_type == "sams":
        

        sams_transforms = transforms.Compose([
            transforms.ToTensor(),
            fn_normalize
        ])
        dataset = SignalDataset(sams_dir,
                                    k = k,
                                    n = n,
                                    transform = sams_transforms)
    elif ref_type == "real_act":
        
        dataset = SignalDatasetRefData(fv,
                                    k = k,
                                    n = n,
                                    mode="activation",
                                    layer_name=layer_name,
                                    transform=fn_normalize)
    elif ref_type == "real_rel":
        
        dataset = SignalDatasetRefData(fv,
                                    k = k,
                                    n = n,
                                    mode="relevance",
                                    layer_name=layer_name,
                                    transform=fn_normalize)
    else:
        raise ValueError(f"Unknown ref_type: {ref_type}, should be [real_act, real_rel, sams]")

    
    ## Compute / Load Distances
    fname_distances = f"{config['dir_precomputed_data']}/dora_data/{config['dataset_name']}_{config['model_name']}_{aggr}/distances/{config['layer_name']}_{ref_type}_{class_id}.pth"
    
    if os.path.isfile(fname_distances):
        D = torch.load(fname_distances)
        print(f"Loading existing distances from {fname_distances}")
    else:
        
        print(len(dataset))
        testloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=2)
        
        A = torch.zeros([k,k,n]).to(device)
        model = modify_model(model, config["layer_name"], aggr="avg")
        with torch.no_grad():
            for i, (x, metainfo) in tqdm.tqdm(enumerate(testloader)):
                x = x.float().to(device)
                acts = model(x)
                r_id = metainfo[0]
                sample_id = metainfo[1]
                A[r_id, :, sample_id] = acts.squeeze(-1).squeeze(-1)

        print(f"Computed activations, shape: {A.shape}")

        A = A.mean(axis = 2)
        D = EA_distance(A, layerwise = True).cpu()
        os.makedirs(os.path.dirname(fname_distances), exist_ok=True)
        torch.save(D, fname_distances)

    print(f"Computed/Loaded distances, shape: {D.shape}")

    ## store distances
    

    ## Automatic outlier detection
    MAX_NUM_CONCEPTS = 3
    MAX_NUM_CONCEPTS_SHOW = 3
    for algorithm in ["umap", "tsne"]:
        data_2d = get_2d_data(D.cpu(), algorithm=algorithm, metric="precomputed")

        # # Find outlier in full distrance matrix (CxC)
        # clf = LocalOutlierFactor(metric = 'precomputed', contamination = 0.01, n_neighbors=20)
        # outliers = clf.fit_predict(D)

        # Find outliers in 2d representation
        outlier_concepts = []
        for k, cids in CONCEPT_ID_COLOR_MAPS.items():
            outlier_concepts += cids

        print(f"Potential outliers: {outlier_concepts}")

        def get_outlier_label(c):
            for k, cids in CONCEPT_ID_COLOR_MAPS.items():
                if c in cids:
                    return k
            return -1
        
        def get_other_label(c):
            for k, cids in IRRELEVANT_CONCEPT_ID_COLOR_MAPS.items():
                if c in cids:
                    return k
            return -1
        
        outlier_labels = [get_outlier_label(x) for x in range(0, len(D))]
        other_labels = [get_other_label(x) for x in range(0, len(D))]

        

        ref_imgs = fv.get_max_reference(outlier_concepts, layer_name, max_mode, (0, n), composite=composite, rf=True,
                                    plot_fn=vis_opaque_img_border)
    
    
        savename = f"{savedir}/{config['dataset_name']}/{config['model_name']}/custom_{config['layer_name']}_{algorithm}_{aggr}_{ref_type}"
        savename += "" if class_id is None else f"_class{class_id}"
        str_concept_ids = "_".join([str(nid) for nid in outlier_concepts[:MAX_NUM_CONCEPTS_SHOW]])
        savename = f"{savename}_{str_concept_ids}"
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        create_plot(data_2d, outlier_labels, other_labels, outlier_concepts, ref_imgs, n, sams_dir, algorithm, f"{savename}.pdf")

def show_sams(sams_dir, concepts, n, axs):
    resize = T.Resize((150, 150))
    for r, concept in enumerate(concepts):
        ax = axs[r]
        grid = make_grid(
            [resize(torch.from_numpy(np.asarray(Image.open(f"{sams_dir}/{concept}_{c}+.jpg"))).permute((2, 0, 1))) for c in range(n)],
            padding=2)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        ax.imshow(grid)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylabel(f"Concept {concept}")

def show_relmax_refimgs(ref_imgs, axs):
    resize = T.Resize((150, 150))
    for r, (concept, imgs) in enumerate(ref_imgs.items()):
        ax = axs[r]
        grid = make_grid(
        [resize(torch.from_numpy(np.asarray(img)).permute((2, 0, 1))) for img in imgs],
            padding=2)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        ax.imshow(grid)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylabel(f"Concept {concept}")

def plot_2d(data, outlier_label, other_label, ax, axis_labels={"x": "UMAP 1", "y": "UMAP 2"}):
    data = pd.DataFrame(data)
    data.columns = ['x', 'y']
    data['outlier_label'] = outlier_label
    data['other_label'] = other_label

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

def create_plot(data, outlier_label, other_label, concepts, ref_imgs, n, sams_dir, algorithm, savename):
    nconcepts = len(concepts)
    nrows = 1 + nconcepts * 2 + 2
    ncols = 1
    base_size = 1.8
    mul_umap = 4
    gap = .2

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(base_size * mul_umap, (nrows-3) * base_size + base_size * mul_umap), 
                            gridspec_kw={'height_ratios':[mul_umap] + [gap] + [1] * nconcepts + [gap] + [1] * nconcepts})

    ax_umap = axs[0]
    plot_2d(data, outlier_label, other_label, ax_umap, axis_labels=
            {
                "x": f"{algorithm.upper()} 1",
                "y": f"{algorithm.upper()} 2"
             }
            )
    axs[1].axis("off"); axs[nconcepts + 2].axis("off")

    axs_actmax_gen = axs[2:2+nconcepts]
    show_sams(sams_dir, concepts, n, axs_actmax_gen)
    axs_actmax_gen[0].set_title("ActMax (Generated)")

    ref_imgs_reduced = {c: ref_imgs[c] for c in concepts}
    axs_relmax = axs[3+nconcepts:]
    show_relmax_refimgs(ref_imgs_reduced, axs_relmax)
    axs_relmax[0].set_title("RelMax")
    fig.savefig(savename, bbox_inches="tight", dpi=300)
    
    ax_umap.axis("off")
    # [ax.axis("off") for ax in axs_relmax]
    fig.savefig(savename[:-4] + "_no_axis" + savename[-4:], bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    main()
