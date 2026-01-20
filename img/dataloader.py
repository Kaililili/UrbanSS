import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os

class Si_Dataset(Dataset):
    def __init__(self, metadata_path, nearest_idx_path, root_dir,
                 height_raw_path, transform=None):
        # satellite image metadata
        self.metadata = pd.read_csv(metadata_path)

        # positive sample index
        self.nearest_idx = np.loadtxt(nearest_idx_path).astype(int)  # shape: (N,)

        # image directory
        self.root_dir = root_dir

        # torchvision transform
        self.transform = transform

        # height raw (N, 64, 64)
        self.height_raw = np.load(height_raw_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        anchor_name = self.metadata.iloc[idx]["satellite_img_name"]
        anchor_img = Image.open(os.path.join(self.root_dir, anchor_name))

        pos_idx = int(self.nearest_idx[idx])
        pos_name = self.metadata.iloc[pos_idx]["satellite_img_name"]
        pos_img = Image.open(os.path.join(self.root_dir, pos_name))


        height_map = self.height_raw[idx]

        height_tensor = torch.tensor(height_map, dtype=torch.float32)

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)

        return anchor_img, pos_img, height_tensor

class Si_Dataset_extract(Dataset):
    def __init__(self, metadata,root_dir, transform = None):
        self.metadata = pd.read_csv(metadata) #list of satellite images
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        path = self.metadata.iloc[idx][0]
        im=Image.open(self.root_dir+path)

        if self.transform:
            sample = self.transform(im)

        return sample,sample