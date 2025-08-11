from matplotlib import pyplot as plt
import numpy as np
import torch

def add_border(x, col, width):
    x_new = torch.ones(3, x.shape[1] + 2 * width, x.shape[2] + 2 * width) * col[:, None, None]
    x_new[:, width:-width, width:-width] = x
    return x_new.type(torch.uint8)

def visualize_dataset(ds, path, start_idx, normalize=True):
    nrows = 4
    ncols = 6
    size = 3

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for i in range(min(nrows * ncols, len(ds))):
        ax = axs[i // ncols][i % ncols]
        idx = start_idx + i
        batch = ds[idx]
        if len(batch) == 2:
            img, y = batch
        else:
            img, y, _ = batch
        if normalize:
            img = np.moveaxis(ds.reverse_normalization(img).numpy(), 0, 2)
        else:
            img = np.moveaxis(img.numpy(), 0, 2)

        # only show blank image without x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(img)
        ax.set_title(ds.map_target_label(y))
    fig.savefig(path)