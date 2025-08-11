
from argparse import ArgumentParser
import os
import torch
from matplotlib import pyplot as plt
import numpy as np
from datasets import load_dataset
from models import get_canonizer, get_fn_model_loader
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.visualization import FeatureVisualization
from zennit.composites import EpsilonPlusFlat
from crp.image import zimage
from torchvision.utils import make_grid
from utils.helper import load_config
from utils._helpers_nmf import overwrite_layer_nmf, run_nmf
from utils.render import vis_opaque_img_border
from torchvision.transforms import Resize
    
def get_parser():

    parser = ArgumentParser()
    
    # ISIC incl bandaid sample
    # parser.add_argument("--sample_ids", default="6,11,531,3661", type=str)

    # Samples where cardiomegaly == 1.0 AND support devices = 1.0

    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--n_concepts", default=64, type=int)
    parser.add_argument("--n_concepts_show", default=10, type=int)
    parser.add_argument("--class_id", default=0, type=int)
    parser.add_argument("--n_refimgs", default=8, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--config_file", 
                        default="config_files/revealing/isic/local/resnet50d_last_conv.yaml")
    parser.add_argument('--savedir', default='plot_files/nmf_plots/')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    config = load_config(args.config_file)
    # sample_ids = [int(i) for i in args.sample_ids.split(",")]
    run_unsupervised_concept_discovery(config, args.n_concepts, args.n_concepts_show, args.n_refimgs, 
                                        args.split, args.class_id, args.batch_size, args.savedir)

def run_unsupervised_concept_discovery(config, n_concepts, n_concepts_show, n_refimgs, split, class_id, batch_size, savedir):

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.get("device", default_device)
    dataset_name = config['dataset_name']
    model_name = config['model_name']
    ckpt_path = config['ckpt_path']

    dataset = load_dataset(config, normalize_data=False)

    splits = {
        "train": dataset.idxs_train,
        "val": dataset.idxs_val,
        "test": dataset.idxs_test,
        }

    dataset = dataset if (split is None) or (split=="all") else dataset.get_subset_by_idxs(splits[split])

    
    # sample_ids = dataset.metadata.loc[(dataset.metadata["Support Devices"] == 1.0) & (dataset.metadata["Cardiomegaly"] == 1.0)].index.values[::100]
    
    model = get_fn_model_loader(model_name=model_name)(n_class=len(dataset.classes), ckpt_path=ckpt_path).to(device).eval()

    layer_name = config["layer_name"]
    
    ## Create NMF model
    mode = "cavs_max"
    scope = [class_id] if class_id is not None else dataset.classes
    H = run_nmf(n_concepts, config, scope, mode)
    pool = "max"
    overwrite_layer_nmf(model, layer_name, H, device, pool)
    layer_name_nmf = f"{layer_name}.1.project"

    ## RUN crp preprocessing for NMF model    
    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()
    layer_map = {layer_name_nmf: cc}
    print(f"using layer {layer_name_nmf}")
    attribution = CondAttribution(model)

    scope_str = "all" if scope is None else "-".join([str(s) for s in scope])
    fv_name = f"{config['dir_precomputed_data']}/crp_files/{dataset_name}_{split}_{model_name}_nmf_{layer_name}-{n_concepts}-{scope_str}"
    print(f"Storing crp results to {fv_name}")
    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.normalize_fn,
                              path=fv_name, cache=None)

    checkdir = f"{fv_name}/ActMax_sum_normed"
    if os.path.isdir(checkdir) and len(os.listdir(checkdir)) > 0:
        print(f"CRP files already exist for {config['config_name']}")
    else:
        while len(dataset) % batch_size == 1:
            batch_size += 1
        print(f"Using batch_size {batch_size}")
        fv.run(composite, 0, len(dataset), batch_size=batch_size)

    ## Plot concepts
    max_mode = "activation"
    ref_imgs = fv.get_max_reference([i for i in range(min(n_concepts, n_concepts_show))], 
                                    layer_name_nmf, max_mode, (0, n_refimgs), composite=composite, rf=True,
                                plot_fn=vis_opaque_img_border)

    savename = f"{savedir}/{config['dataset_name']}/{config['model_name']}/nmf_{layer_name}_{n_concepts}concepts_{scope_str}.png"
    store_plot(ref_imgs, n_refimgs, savename)



def store_plot(ref_imgs, n_ref_imgs, savename):
    s = 1.6
    n_concepts = len(ref_imgs)
    fig, axs = plt.subplots(n_concepts, 1, squeeze=False, figsize=(s * n_ref_imgs, s* n_concepts))
    resize = Resize((150, 150))
    for cid in range(n_concepts):
        ax = axs[cid][0]
        grid = grid = make_grid([torch.from_numpy(np.asarray(resize(img))).permute((2, 0, 1)) for img in ref_imgs[cid]])
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        ax.imshow(grid)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylabel(f"Concept {cid}")

    os.makedirs(os.path.dirname(savename), exist_ok=True)
    fig.savefig(savename, bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    main()
