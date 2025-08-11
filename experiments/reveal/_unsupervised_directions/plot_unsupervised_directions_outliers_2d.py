from argparse import ArgumentParser

import numpy as np
from datasets import DATASET_CLASSES, load_dataset
from models import get_fn_model_loader, get_canonizer
from utils.dimensionality_reduction import get_2d_data

from utils.helper import load_config, none_or_int, none_or_str
import os
import torch
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
from sklearn.neighbors import LocalOutlierFactor
from zennit.composites import EpsilonPlusFlat
from torch.utils.data import DataLoader
from utils._helpers_nmf import overwrite_layer_nmf, run_nmf
from utils.render import vis_opaque_img_border

## Automatic outlier detection
MAX_NUM_CONCEPTS = 3
MAX_NUM_CONCEPTS_SHOW = 3

def get_parser():
    parser = ArgumentParser(
        description='Plot NMF 2D embedding with outliers.', )

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_concepts', default=64, type=int)
    parser.add_argument('--class_id', default=0, type=none_or_int)
    parser.add_argument('--direction_ids', type=str, 
                        default=None)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--aggr', default="max", type=none_or_str)
    parser.add_argument('--max_mode', default="relevance", type=str)
    parser.add_argument('--savedir', default="plot_files/directions_outliers", type=str)
    parser.add_argument('--config_file',
                        # default="config_files/revealing/isic/local/vgg16_features.29.yaml")
                        default="config_files/revealing/isic/local/resnet50d_last_conv.yaml")
   

    return parser

def main():
    args = get_parser().parse_args()
    config = load_config(args.config_file)
    plot_unsupervised_concept_embedding(config, args.n_concepts, args.direction_ids, 
                                        args.batch_size, args.split, args.class_id, args.savedir)


def cosinesim_matrix(X: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(X) @ torch.nn.functional.normalize(X).t()

def plot_unsupervised_concept_embedding(config, n_concepts, direction_ids, batch_size, 
                                        split, class_id, savedir):
    
    model_name = config["model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(DATASET_CLASSES[config["dataset_name"]].classes)
    model = get_fn_model_loader(model_name)(n_class=n_classes,
                                            ckpt_path=config["ckpt_path"]
                                            ).to(device).eval()

    
    dataset = load_dataset(config, normalize_data=True)
    ref_dataset = load_dataset(config, normalize_data=False)

    splits = {
        "train": dataset.idxs_train,
        "val": dataset.idxs_val,
        "test": dataset.idxs_test,
        }

    dataset_name = config['dataset_name']
    dataset = dataset if (split is None) or (split=="all") else dataset.get_subset_by_idxs(splits[split])
    ref_dataset = ref_dataset if (split is None) or (split=="all") else ref_dataset.get_subset_by_idxs(splits[split])
    layer_name = config["layer_name"]

    ## Create NMF model
    mode = "cavs_max"
    scope = [class_id] if class_id is not None else dataset.classes
    H = run_nmf(n_concepts, config, scope, mode)
    pool = "max"
    overwrite_layer_nmf(model, layer_name, H, device, pool)
    layer_name_nmf = f"{layer_name}.1.project"

    scope_str = "all" if scope is None else "-".join([str(s) for s in scope])
    fv_name = f"{config['dir_precomputed_data']}/crp_files/{dataset_name}_{split}_{model_name}_nmf_{layer_name}-{n_concepts}-{scope_str}"
    assert os.path.isdir(f"{fv_name}/ActMax_sum_normed"), f"Run CRP preprocessing first (in run_unsupervised_concept_discovery)"

    
    
    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()
    layer_map = {layer_name_nmf: cc}
    print(f"using layer {layer_name_nmf}")
    attribution = CondAttribution(model)
    
    fv = FeatureVisualization(attribution, ref_dataset, layer_map, preprocess_fn=ref_dataset.normalize_fn,
                              path=fv_name, cache=None)
    
    ## Compute / Load Distances
    fname_similarities = f"{config['dir_precomputed_data']}/concept_clustering_nmf/{config['dataset_name']}_{split}_{config['model_name']}/similarities/{config['layer_name']}_{n_concepts}concepts_{class_id}.pth"
    
    if not os.path.isfile(fname_similarities):
        print(len(dataset))
        dl = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=2)
        
        layer_names = [layer_name_nmf]

        activations = []
        # with torch.no_grad():
        
        for x, y in tqdm.tqdm(dl):
            condition = [{"y": _y} for _y in y]
            attr = attribution(x.to(device).requires_grad_(), condition, composite, record_layer=layer_names)
            activations.append(attr.activations[layer_name_nmf].detach().cpu())

        activations = torch.cat(activations)

        os.makedirs(os.path.dirname(fname_similarities), exist_ok=True)

        similarities = {}
        similarities[f"act_nmf_sim"] = cosinesim_matrix(activations.transpose(0,1))

        torch.save(similarities, fname_similarities)

    sim = torch.load(fname_similarities)["act_nmf_sim"]
    D = (1-sim).clamp(min=0)
    print(f"Computed/Loaded distances, shape: {D.shape}")

    for algorithm in ["umap", "tsne"]:
        data_2d = get_2d_data(D, algorithm=algorithm, metric="precomputed")

        # Find outliers in 2d representation
        clf = LocalOutlierFactor(contamination = 0.01, n_neighbors=20)
        _ = clf.fit_predict(data_2d)
        p = torch.tensor(clf.negative_outlier_factor_)
        thresh = -1.0
        top_outlier_idx = p.argsort()
        outlier_concepts = top_outlier_idx[p[top_outlier_idx] < thresh][:MAX_NUM_CONCEPTS].numpy()
        outlier_concepts = [int(nid) for nid in direction_ids.split(",")] if direction_ids is not None else outlier_concepts
        print(f"Potential outliers: {outlier_concepts}")
        
        def get_outlier_label(c):
            if c in outlier_concepts:
                return 1
            else:
                return 0
            
        outlier_labels = [get_outlier_label(x) for x in range(0, len(D))]
        
        
        

        ref_imgs = fv.get_max_reference(outlier_concepts[:MAX_NUM_CONCEPTS_SHOW], layer_name_nmf, "activation", (0, 5), composite=composite, rf=True,
                                    plot_fn=vis_opaque_img_border)
    

        savename = f"{savedir}/{config['dataset_name']}/{config['model_name']}/{config['layer_name']}_{n_concepts}concepts_{algorithm}"
        savename += "" if class_id is None else f"_class{class_id}"
        if direction_ids is not None:
            str_concept_ids = "_".join([str(nid) for nid in outlier_concepts[:MAX_NUM_CONCEPTS_SHOW]])
            savename = f"{savename}_{str_concept_ids}"
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        create_plot(data_2d, outlier_labels, outlier_concepts[:MAX_NUM_CONCEPTS_SHOW], ref_imgs, algorithm, f"{savename}.png")

def show_actmax_refimgs(ref_imgs, axs):
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

def plot_2d(data, label, ax, axis_labels={"x": "UMAP 1", "y": "UMAP 2"}):
    data = pd.DataFrame(data)
    data.columns = ['x', 'y']
    data['label'] = label
    
    sns.scatterplot(
        data=data[data.label == 0], x="x", y="y", 
        hue = 'label', palette=sns.color_palette()[:1],
        s=50, alpha = 0.25,
        ax=ax, legend=False
    )

    sns.scatterplot(
        data=data[data.label != 0], x="x", y="y", 
        hue = 'label', palette=sns.color_palette()[1:len(np.unique(data['label'].values))],
        s=50, alpha = 1,
        ax=ax, legend=False
)

    ax.set(xlabel=axis_labels["x"], ylabel=axis_labels["y"])
    sns.despine()

def create_plot(data, label, concepts, ref_imgs, algorithm, savename):
    nconcepts = len(concepts)
    nrows = 1 + nconcepts + 1
    ncols = 1
    base_size = 1.8
    mul_umap = 4
    gap = .2

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(base_size * mul_umap, (nrows-3) * base_size + base_size * mul_umap), 
                            gridspec_kw={'height_ratios':[mul_umap] + [gap] + [1] * nconcepts})

    ax_umap = axs[0]
    plot_2d(data, label, ax_umap, axis_labels=
            {
                "x": f"{algorithm.upper()} 1",
                "y": f"{algorithm.upper()} 2"
             }
            )
    axs[1].axis("off"); 

    ref_imgs_reduced = {c: ref_imgs[c] for c in concepts}
    axs_relmax = axs[2:2+nconcepts]
    
    show_actmax_refimgs(ref_imgs_reduced, axs_relmax)
    axs_relmax[0].set_title("ActMax")
    fig.savefig(savename, bbox_inches="tight")
    
if __name__ == "__main__":
    main()
