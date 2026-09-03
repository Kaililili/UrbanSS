import argparse
import random
import rasterio
from pyproj import Transformer
from shapely.geometry import box
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point
from rtree import index
from rasterio.windows import from_bounds
from collections import defaultdict
from rasterio.mask import mask
from shapely.geometry import mapping
import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HeightDataset(Dataset):
    def __init__(self, data_list, transform):

        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        tensor = item["tensor"]

        if self.transform:
            data1 = self.transform(tensor)
            data2 = self.transform(tensor)

        return data1, data2

class HeightCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        z = F.normalize(z, dim=1)
        return z

def info_nce_loss(z1, z2, temperature=0.5):

    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)   # (2B, D)

    sim = torch.matmul(z, z.T) / temperature
    sim = sim - torch.eye(2*B, device=z.device) * 1e9

    pos = torch.sum(z1 * z2, dim=-1) / temperature
    pos = torch.cat([pos, pos], dim=0)  # (2B)

    # InfoNCE Loss
    loss = -pos + torch.logsumexp(sim, dim=1)
    return loss.mean()

def which_region_fast(lon, lat):
    pt = Point(lon, lat)
    candidates = spatial_index.intersection((lon, lat, lon, lat))
    for idx in candidates:
        if region_boxes[idx].contains(pt):
            return region_ids[idx]
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Building height CNN training")
    parser.add_argument("--city", type=str, default="Shenzhen", help="city name")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    args = parser.parse_args()

    city = args.city
    region_path = f""
    region_df = pd.read_csv(region_path)

    name2id = {name: idx for idx, name in enumerate(region_df["satellite_img_name"].unique())}
    region_boxes = []
    region_ids = []
    spatial_index = index.Index()

    for i, row in region_df.iterrows():
        ll_lat, ll_lon = map(float, row['WGS84_lower_left'].strip("()").split(','))
        ur_lat, ur_lon = map(float, row['WGS84_upper_right'].strip("()").split(','))

        rect = box(min(ll_lon, ur_lon), min(ll_lat, ur_lat),
                   max(ll_lon, ur_lon), max(ll_lat, ur_lat))
        region_boxes.append(rect)
        name = row['satellite_img_name']
        region_id = name2id[name]
        region_ids.append(region_id)
        spatial_index.insert(i, rect.bounds)


    min_lon = min(region_df['WGS84_lower_left'].apply(lambda s: float(s.strip("()").split(',')[1])))
    min_lat = min(region_df['WGS84_lower_left'].apply(lambda s: float(s.strip("()").split(',')[0])))
    max_lon = max(region_df['WGS84_upper_right'].apply(lambda s: float(s.strip("()").split(',')[1])))
    max_lat = max(region_df['WGS84_upper_right'].apply(lambda s: float(s.strip("()").split(',')[0])))

    bounds = box(min_lon, min_lat, max_lon, max_lat)

    height_tif_path = ""

    region_boxes = []
    for i, row in region_df.iterrows():
        ll_lat, ll_lon = map(float, row['WGS84_lower_left'].strip("()").split(','))
        ur_lat, ur_lon = map(float, row['WGS84_upper_right'].strip("()").split(','))
        geom = box(min(ll_lon, ur_lon), min(ll_lat, ur_lat), max(ll_lon, ur_lon), max(ll_lat, ur_lat))
        name = row['satellite_img_name']
        region_id = name2id[name]
        region_boxes.append({'region_id': region_id, 'geometry': geom})

    regions_gdf = gpd.GeoDataFrame(region_boxes, crs='EPSG:4326')

    raster = rasterio.open(height_tif_path)
    regions_utm = regions_gdf.to_crs(raster.crs)
    cnn_inputs = []
    for idx, row in regions_utm.iterrows():
        geom = row['geometry']
        try:
            out_image, _ = mask(raster, [mapping(geom)], crop=True)
            masked_data = out_image[0]
            masked_data = np.where(masked_data <= 0, 0, masked_data)

            masked_resized = resize(masked_data, (64, 64), mode="constant", anti_aliasing=True)
            tensor_input = torch.tensor(masked_resized, dtype=torch.float32).unsqueeze(0)
            cnn_inputs.append({"region_id": row["region_id"], "tensor": tensor_input})

        except Exception as e:
            print(f"Region {row['region_id']} error: {e}")

    contrastive_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.ConvertImageDtype(torch.float32),
    ])

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    dataset = HeightDataset(cnn_inputs, contrastive_transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = HeightCNN(embedding_dim=128).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    model.train()
    for epoch in range(100):
        total_loss = 0
        for data1, data2 in dataloader:
            data1 = data1.to(device)
            data2 = data2.to(device)
            z1, z2 = model(data1), model(data2)

            loss = info_nce_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Loss = {avg_loss:.4f}")

    model.eval()
    all_embeddings = []
    region_ids = []

    with torch.no_grad():
        for item in cnn_inputs:
            region_ids.append(item["region_id"])
            tensor_input = item["tensor"].to(device)
            embedding = model(tensor_input.unsqueeze(0))
            all_embeddings.append(embedding.squeeze(0).cpu().numpy())

    # also save the 64x64 height crops
    # (used by img/train.py for hard-negative mining; row order matches the region list)
    height_crops = np.stack([item['tensor'].squeeze(0).numpy() for item in cnn_inputs], axis=0)
    np.save("height_crop_raw.npy", height_crops)

    all_embeddings = np.stack(all_embeddings, axis=0)
    np.save("height_embedding.npy", all_embeddings)



