import random
from pathlib import Path

import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from datasets.waterbirds.generate_waterbirds import generate_waterbirds
from datasets.base_dataset import BaseDataset


def create_waterbirds_dataset(data_paths, p_artifact_landbirds, p_artifact_waterbirds, cub_places_path, normalize_data=True, image_size=224, **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)

    # Create Dataset
    cub_dir = f"{cub_places_path[0]}/CUB_200_2011"
    places_dir = f"{cub_places_path[0]}/data_256"
    
    if p_artifact_landbirds == 0.0 and p_artifact_waterbirds == 0.0: 
        dataset_name = 'waterbirds_clean'
    elif p_artifact_landbirds == 1.0 and p_artifact_waterbirds == 1.0:
        dataset_name = 'waterbirds_attacked'
    else:
        dataset_name = f'waterbirds_p_artifact_landbirds{p_artifact_landbirds}_waterbirds{p_artifact_waterbirds}'

    # Construct the full path
    dataset_dir = os.path.join(data_paths[0], dataset_name)

    # Check if the directory exists
    if os.path.isdir(dataset_dir):
        print(f"The directory '{dataset_dir}' already exists.")
        output_subfolder = dataset_dir
    else:
        print(f"The directory '{dataset_dir}' does not exist. Creating it now...")
        seed=42
        output_subfolder = generate_waterbirds(
            p_artifact_landbirds, p_artifact_waterbirds, data_paths[0], cub_dir, places_dir, dataset_name, seed=seed
        )
    
    p_artifacts = [p_artifact_landbirds, p_artifact_waterbirds]

    return output_subfolder, transform, p_artifacts, cub_places_path, image_size, kwargs

def get_waterbirds_dataset(data_paths, p_artifact_landbirds, p_artifact_waterbirds, cub_places_path, normalize_data=True, image_size=224, **kwargs):
    output_subfolder, transform, p_artifacts, cub_places_path, image_size, kwargs = create_waterbirds_dataset(data_paths, p_artifact_landbirds, 
                                                                                        p_artifact_waterbirds, cub_places_path, normalize_data, image_size, **kwargs)
    return WaterbirdsDataset(output_subfolder, transform=transform, p_artifacts=p_artifacts, cub_places_path=cub_places_path,
                            image_size=image_size, augmentation=None, **kwargs)


class WaterbirdsDataset(BaseDataset):
    classes = ['landbird', 'waterbird']
    
    def __init__(self, path, cub_places_path, p_artifacts, transform, image_size, augmentation, **kwargs):

        super().__init__(data_paths=[path], transform=transform, augmentation=augmentation)

        self.path = Path(path)
        self.cub_places_path = cub_places_path
        self.p_artifacts = p_artifacts
        self.transform = transform
        self.image_size = image_size
        self.metadata = None
        self.load_metadata(self.path / 'metadata.csv')
        self.targets = self.metadata['y'].values
        self.idxs_train, self.idxs_val, self.idxs_test = self.do_train_val_test_split(.1, .1)
        self.classes = ['landbird', 'waterbird']
        self.classes_id = [0, 1]
        self.groups = self.get_group_list()

        dist = np.array([len(self.metadata) - self.metadata['y'].sum(), self.metadata['y'].sum()])
        self.weights = self.compute_weights(dist)

        self.sample_ids_by_artifact = self.get_ids_by_artifact() # Waterbackground & Landbackground
        self.artifact_ids = [sample_id for _, sample_ids in self.sample_ids_by_artifact.items() for sample_id
                                        in sample_ids]
        self.clean_sample_ids = list(set(np.arange(len(self))) - set(self.artifact_ids)) # Should be empty

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        img = Image.open(self.path / row['img_filename'])
        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(row['y'])

    def __len__(self):
        return len(self.metadata)

    def get_subset_by_idxs(self, idxs):
        subset = super().get_subset_by_idxs(idxs)
        subset.metadata = subset.metadata.iloc[idxs].reset_index(drop=True)
        return subset

    def load_metadata(self, metadata_path):
        self.metadata = pd.read_csv(metadata_path)

    def get_target(self, index):
        return self.targets[index]

    def map_target_label(self, target):
        if target == 1:
            return 'Waterbird'
        else:
            return 'Landbird'

    def get_ids_by_artifact(self):
        artifact_ids = {}
        ids_water = []
        ids_land = []
        for idx, row in self.metadata.iterrows():
            if row['place'] == 1: # Waterplace
                ids_water.append(idx)
            elif row['place'] == 0: # Landplace
                ids_land.append(idx)
        artifact_ids['waterbackground'] = ids_water
        artifact_ids['landbackground'] = ids_land
        artifact_ids['background'] = ids_water + ids_land

        return artifact_ids
    
    def get_group_list(self):
        """
        Groups:
        - 0: landbird + landplace
        - 1: landbird + waterplace
        - 2: waterbird + landplace
        - 3: waterbird + waterplace
        

        return list of groups as tensor
        """
        groups = []
        for idx, row in self.metadata.iterrows():
            if row['y'] == 0 and row['place'] == 0:
                groups.append(0)
            elif row['y'] == 0 and row['place'] == 1:
                groups.append(1)
            elif row['y'] == 1 and row['place'] == 0:
                groups.append(2)
            elif row['y'] == 1 and row['place'] == 1:
                groups.append(3)
        return torch.tensor(groups)
    
    def get_class_id_by_name(self, class_name):
        return 0 if class_name == 'landbird' else 1
