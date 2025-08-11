from argparse import ArgumentParser
from datasets import DATASET_CLASSES, DATASET_NORM_PARAMS
from models import get_fn_model_loader
from utils.dora.dora import EA_distance, SignalDataset
from utils.dora.model import get_dim, modify_model

from utils.helper import load_config
import os
import torch
import torchvision.transforms as transforms
import tqdm
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_parser():
    parser = ArgumentParser(
        description='Generate sAMS for DORA Analysis.', )

    parser.add_argument('--n', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--aggr', default="avg", type=str)
    parser.add_argument('--savedir', default="plot_files/dora", type=str)
    parser.add_argument('--config_file',
                        default="config_files/revealing/imagenet/local/vgg16_features.28.yaml", )
   

    return parser

def main():
    args = get_parser().parse_args()
    config = load_config(args.config_file)
    plot_umap(config, args.n, args.aggr, args.batch_size, args.savedir)


def plot_umap(config, n, aggr, batch_size, savedir):
    sams_dir = f"{config['dir_precomputed_data']}/dora_data/{config['dataset_name']}_{config['model_name']}_{aggr}/sAMS/{config['model_name']}_{config['layer_name']}/"

    mean, std = DATASET_NORM_PARAMS[config['dataset_name']]

    sams_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                            std=std)
    ])

    model_name = config["model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(DATASET_CLASSES[config["dataset_name"]].classes)
    model = get_fn_model_loader(model_name)(n_class=n_classes,
                                            ckpt_path=config["ckpt_path"]
                                            ).to(device).eval()

    model = modify_model(model, config["layer_name"], aggr="avg")
    k = get_dim(model, config["img_size"], device)

    dataset = SignalDataset(sams_dir,
                            k = k,
                            n = n,
                            transform = sams_transforms)
    print(len(dataset))
    testloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=2)
    
    A = torch.zeros([k,k,n]).to(device)

    with torch.no_grad():
        for i, (x, metainfo) in tqdm.tqdm(enumerate(testloader)):
            x = x.float().to(device)
            acts = model(x)
            r_id = metainfo[0]
            sample_id = metainfo[1]
            A[r_id, :, sample_id] = acts.squeeze(-1).squeeze(-1)

    print(f"Computed activations, shape: {A.shape}")

    A = A.mean(axis = 2)
    D = EA_distance(A, layerwise = True)

    print(f"Computed distances, shape: {D.shape}")

    umap_op = umap.UMAP(metric='precomputed')
    data_umap = umap_op.fit_transform(D.cpu())
    savename = f"{savedir}/{config['dataset_name']}/{config['model_name']}/{config['layer_name']}_{aggr}.png"
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    create_plot(data_umap, savename)

def create_plot(data, savename):
    f, ax = plt.subplots(figsize=(8, 8))

    # color = [1 if x in CH_resps else 0 for x i
    # n range(0, 512)]

    data = pd.DataFrame(data)
    data.columns = ['x', 'y']
    # data['label'] = color

    ax = sns.scatterplot(
        data=data, x="x", y="y", 
    #     hue = 'label', 
        s=50, alpha = 0.65, legend = False,
    )

    ax.set(xlabel='UMAP 1', ylabel='UMAP 2')
    sns.despine()
    f.savefig(savename)
    
if __name__ == "__main__":
    main()
