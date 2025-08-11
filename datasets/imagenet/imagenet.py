import copy
import glob
import json
from collections import Counter
import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from datasets.base_dataset import BaseDataset
import random

imagenet_augmentation = T.Compose([
    T.RandomHorizontalFlip(p=0.25),
    T.RandomVerticalFlip(p=0.25),
    T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.25),
    T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=0.25),
    T.RandomApply(transforms=[T.ColorJitter(brightness=0.1, saturation=0.1, hue=0.1)], p=0.25)
])

NORM_PARAMS_IMAGENET = (torch.Tensor((0.485, 0.456, 0.406)),
                        torch.Tensor((0.229, 0.224, 0.225)))


def get_imagenet_dataset(data_paths,
                         normalize_data=True,
                         image_size=224,
                         artifact_ids_file=None,
                         label_map_path=None,
                         classes=None,
                         subset=None,
                         only_val=False,
                         **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]
    if normalize_data:
        fns_transform.append(T.Normalize(*NORM_PARAMS_IMAGENET))
    transform = T.Compose(fns_transform)
    return ImageNetDataset(data_paths, transform=transform, augmentation=imagenet_augmentation,
                           artifact_ids_file=artifact_ids_file, label_map_path=label_map_path,
                           classes=classes, subset=subset, only_val=only_val)


def extract_wnid(path):
    return path.split("/")[-1]


class ImageNetDataset(BaseDataset):
    # Default allowed (restricted) class names
    classes = ['Dog', 'Cat', 'Frog', 'Turtle', 'Bird', 'Primate', 'Fish', 'Crab', 'Insect']
    
    # Mapping from new class names to lists of normal numeric ids (as strings)
    restricted_classes_mapping = {
        "Dog": [str(i) for i in range(151, 269)],
        "Cat": [str(i) for i in range(281, 286)],
        "Frog": [str(i) for i in range(30, 33)],
        "Turtle": [str(i) for i in range(33, 38)],
        "Bird": [str(i) for i in range(80, 101)],
        "Primate": [str(i) for i in range(365, 383)],
        "Fish": [str(i) for i in range(389, 398)],
        "Crab": [str(i) for i in range(118, 122)],
        "Insect": [str(i) for i in range(300, 320)]
    }
    
    def __init__(self,
                 data_paths,
                 transform=None,
                 augmentation=None,
                 artifact_ids_file=None,
                 label_map_path=None,
                 classes=None,
                 subset=None,
                 only_val=False):
        # Use provided classes or default to our restricted classes.
        if classes is None:
            classes = self.classes
        self.classes = classes  # allowed new class names

        super().__init__(data_paths, transform, augmentation, artifact_ids_file)
        
        # Load the original label map from JSON.
        # The label map should map wnids to a dict containing at least a "label" field.
        assert label_map_path is not None, "label_map_path is required for ImageNetDataset"
        with open(label_map_path) as file:
            orig_label_map = json.load(file)
        
        # Define the paths for the train and val splits.
        path_train = f"{data_paths[0]}/train"
        path_val = f"{data_paths[0]}/val"
        
        # Choose the base path from which to read samples.
        # If only_val is True, use the official val folder for all splits.
        self.only_val = only_val
        base_path = path_val if only_val else path_train
        
        # Read all samples from the base path.
        all_samples = self.read_samples(base_path)
        
        # Map each sample’s wnid to a new class name.
        # Store samples as (image_path, original_wnid, new_class)
        mapped_samples = []
        for sample in all_samples:
            img_path, wnid = sample
            if wnid not in orig_label_map:
                continue
            numeric_id = orig_label_map[wnid].get("label")
            if numeric_id is None:
                continue
            new_class = None
            # Convert numeric_id to string to match the mapping lists.
            for class_name, id_list in self.restricted_classes_mapping.items():
                if str(numeric_id) in id_list:
                    new_class = class_name
                    break
            if new_class is not None and new_class in self.classes:
                mapped_samples.append((img_path, wnid, new_class))
        random.seed(0)
        random.shuffle(mapped_samples)
        self.samples = mapped_samples
        
        # Build a new label map: assign each allowed new class a unique integer label.
        self.label_map = {cls: {"label": torch.tensor(idx).long()}
                          for idx, cls in enumerate(self.classes)}
        
        # Compute class distribution for weight computation.
        counts = Counter([new_class for _, _, new_class in self.samples])
        print("Counts:", counts)
        dist = torch.Tensor([counts[cls] for cls in self.classes])
        
        self.mean = NORM_PARAMS_IMAGENET[0]
        self.var = NORM_PARAMS_IMAGENET[1]
        self.weights = self.compute_weights(dist)
        
        # Create train/val/test splits.
        if only_val:
            # Use the official val set for all splits.
            self.idxs_train, self.idxs_val, self.idxs_test = self.do_train_val_test_split(0.1, 0.1)
        else:
            # Split the training set (from the train folder) into train/val.
            self.idxs_train, self.idxs_val, self.idxs_test = self.do_train_val_test_split(0.1, 0)
            # Use the official val folder as the test set.
            num_samples_before = len(self.samples)
            val_samples = self.read_samples(path_val)
            mapped_val_samples = []
            for sample in val_samples:
                img_path, wnid = sample
                if wnid not in orig_label_map:
                    continue
                numeric_id = orig_label_map[wnid].get("label")
                if numeric_id is None:
                    continue
                new_class = None
                for class_name, id_list in self.restricted_classes_mapping.items():
                    if str(numeric_id) in id_list:
                        new_class = class_name
                        break
                if new_class is not None and new_class in self.classes:
                    mapped_val_samples.append((img_path, wnid, new_class))
            self.samples += mapped_val_samples
            self.idxs_test = np.arange(num_samples_before, len(self.samples))
        
        self.sample_ids_by_artifact = self.get_sample_ids_by_artifact()
        self.all_artifact_sample_ids = [
            sample_id 
            for _, sample_ids in self.sample_ids_by_artifact.items() 
            for sample_id in sample_ids
        ]
        self.clean_sample_ids = list(set(np.arange(len(self))) - set(self.all_artifact_sample_ids))

        self.targets = [self.get_target(i) for i in range(len(self))]
    
    def read_samples(self, path, classes=None):
        """
        Reads samples from the given path and returns a list of tuples (image_path, wnid).
        Mapping to new class names is performed in __init__.
        """
        samples = []
        for subdir in sorted(glob.glob(f"{path}/*")):
            wnid = extract_wnid(subdir)
            for img_path in sorted(glob.glob(f"{subdir}/*.JPEG")):
                samples.append((img_path, wnid))
        return samples
    
    def get_all_ids(self):
        return [os.path.basename(sample[0]) for sample in self.samples]
    
    def __len__(self):
        return len(self.samples)
    
    def get_target(self, idx):
        # Now each sample is (img_path, original_wnid, new_class)
        _, _, new_class = self.samples[idx]
        return self.label_map[new_class]["label"]
    
    def get_class_id_by_name(self, class_name):
        return self.label_map[class_name]["label"]
    
    def __getitem__(self, idx):
        # Unpack the tuple with original wnid preserved.
        img_path, orig_wnid, new_class = self.samples[idx]
        x = Image.open(img_path).convert('RGB')
        x = self.transform(x) if self.transform else x
        if self.do_augmentation and self.augmentation:
            x = self.augmentation(x)
        y = self.label_map[new_class]["label"]
        return x, y
    
    def get_subset_by_idxs(self, idxs):
        subset = copy.deepcopy(self)
        subset.samples = [subset.samples[i] for i in idxs]
        return subset

    def map_target_label(self, y):
        for class_name, label in self.label_map.items():
            if label["label"] == y:
                return class_name
        return None
