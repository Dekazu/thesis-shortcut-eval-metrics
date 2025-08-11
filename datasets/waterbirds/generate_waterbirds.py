import os
import numpy as np
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm

def crop_and_resize(source_img, target_img):
    """
    Make source_img exactly the same as target_img by expanding/shrinking and
    cropping appropriately.
    """
    source_width, source_height = source_img.size
    target_width, target_height = target_img.size

    # If source is too small, resize up then re-crop
    if source_width < target_width or source_height < target_height:
        width_resize = (
            target_width,
            int((target_width / source_width) * source_height)
        )
        if width_resize[0] >= target_width and width_resize[1] >= target_height:
            source_resized = source_img.resize(width_resize, Image.Resampling.LANCZOS)
        else:
            height_resize = (
                int((target_height / source_height) * source_width),
                target_height
            )
            source_resized = source_img.resize(height_resize, Image.Resampling.LANCZOS)
        return crop_and_resize(source_resized, target_img)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height

    if source_aspect > target_aspect:
        # Crop left/right
        new_w = int(target_aspect * source_height)
        offset = (source_width - new_w) // 2
        crop_box = (offset, 0, offset + new_w, source_height)
    else:
        # Crop top/bottom
        new_h = int(source_width / target_aspect)
        offset = (source_height - new_h) // 2
        crop_box = (0, offset, source_width, offset + new_h)

    cropped = source_img.crop(crop_box)
    return cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)

def combine_and_mask(img_new, mask, img_black):
    """
    Combine img_new, mask, and img_black based on the mask.
    """
    img_resized = crop_and_resize(img_new, img_black)
    img_np = np.asarray(img_resized)
    img_masked = np.around(img_np * (1 - mask)).astype(np.uint8)
    combined_np = np.asarray(img_black) + img_masked
    return Image.fromarray(combined_np)

def generate_waterbirds(
    p_artifact_landbirds,
    p_artifact_waterbirds,
    output_dir,
    cub_dir,
    places_dir,
    dataset_name='Waterbirds',
    seed=42
):
    # ---- initialization ----
    val_frac = 0.2
    np.random.seed(seed)
    random.seed(seed)

    target_places = [
        ['bamboo_forest', 'forest/broadleaf', 'forest_path'],  # land
        ['ocean', 'lake/natural', 'beach']                    # water
    ]

    # ---- load CUB metadata & label y ----
    images_path = os.path.join(cub_dir, 'images.txt')
    df = pd.read_csv(
        images_path, sep=' ', header=None,
        names=['img_id', 'img_filename'], index_col='img_id'
    )
    species = np.unique([
        fn.split('/')[0].split('.')[1].lower()
        for fn in df['img_filename']
    ])
    water_birds_list = [
        'Albatross','Auklet','Cormorant','Frigatebird','Fulmar','Gull','Jaeger',
        'Kittiwake','Pelican','Puffin','Tern','Gadwall','Grebe','Mallard',
        'Merganser','Guillemot','Pacific_Loon'
    ]
    water_birds = {s.lower(): 0 for s in species}
    for s in species:
        for wb in water_birds_list:
            if wb.lower() in s.lower():
                water_birds[s] = 1
    df['y'] = [
        water_birds[fn.split('/')[0].split('.')[1].lower()]
        for fn in df['img_filename']
    ]

    # ---- original train/test split → new train/val/test ----
    tts = pd.read_csv(
        os.path.join(cub_dir, 'train_test_split.txt'),
        sep=' ', header=None, names=['img_id','split'], index_col='img_id'
    )
    df = df.join(tts, on='img_id')
    test_ids  = df.loc[df['split'] == 0].index
    train_ids = df.loc[df['split'] == 1].index
    val_ids   = np.random.choice(
        train_ids,
        size=int(round(val_frac * len(train_ids))),
        replace=False
    )
    df.loc[train_ids, 'split'] = 0
    df.loc[val_ids,   'split'] = 1
    df.loc[test_ids,  'split'] = 2

    # ---- inject spurious correlation ----
    df['place'] = 0
    for split_idx, ids in enumerate([train_ids, val_ids, test_ids]):
        for y in (0,1):
            p = p_artifact_landbirds if y == 0 else p_artifact_waterbirds
            y_ids = df.loc[ids].loc[df['y'] == y].index
            n_pos = int(round(p * len(y_ids)))
            chosen = np.random.choice(y_ids, size=n_pos, replace=False)
            df.loc[chosen, 'place'] = 1

    # (optional) print balances
    for split,label in [(0,'train'),(1,'val'),(2,'test')]:
        sub = df[df['split']==split]
        print(f"{label}: waterbird fraction = {sub['y'].mean():.3f}")
        for y in (0,1):
            for c in (0,1):
                cnt = ((sub['y']==y)&(sub['place']==c)).sum()
                total = (sub['y']==y).sum() or 1
                print(f"  y={y},c={c}: {cnt/total:.3f} ({cnt})")

    # ---- build & shuffle background pools once ----
    bg_pools = {}
    for idx, place_list in enumerate(target_places):
        pool = []
        for tp in place_list:
            dirpath = os.path.join(places_dir, tp[0], tp)
            for fn in sorted(os.listdir(dirpath)):
                if fn.lower().endswith('.jpg'):
                    pool.append(f"/{tp[0]}/{tp}/{fn}")
        random.shuffle(pool)
        bg_pools[idx] = pool

    # ---- assign backgrounds per split, without reuse ----
    for split_idx, split_label in enumerate(['train','val','test']):
        print(f"Assigning backgrounds for {split_label}…")
        for cls in (0, 1):
            mask     = (df['split']==split_idx) & (df['place']==cls)
            n_needed = mask.sum()

            # take the first n_needed from the pool
            sel = bg_pools[cls][:n_needed]
            df.loc[mask, 'place_filename'] = sel

            # remove those used backgrounds so they won't be reused
            bg_pools[cls] = bg_pools[cls][n_needed:]

    # ---- write metadata & composite images ----
    out_folder = os.path.join(output_dir, dataset_name)
    os.makedirs(out_folder, exist_ok=True)
    df.to_csv(os.path.join(out_folder, 'metadata.csv'))

    for i in tqdm(df.index, desc="Compositing images"):
        img_path = os.path.join(cub_dir, 'images', df.loc[i,'img_filename'])
        seg_path = os.path.join(
            cub_dir, 'segmentations',
            df.loc[i,'img_filename'].replace('.jpg','.png')
        )

        img = Image.open(img_path).convert('RGB')
        seg = np.asarray(Image.open(seg_path).convert('RGB')) / 255

        bg_rel = df.loc[i,'place_filename'][1:]
        bg = Image.open(os.path.join(places_dir, bg_rel)).convert('RGB')

        bird_only = Image.fromarray(
            np.around(np.asarray(img) * seg).astype(np.uint8)
        )
        combined  = combine_and_mask(bg, seg, bird_only)

        out_path = os.path.join(out_folder, df.loc[i,'img_filename'])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        combined.save(out_path)

    return out_folder
