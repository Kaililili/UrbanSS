import argparse
import torch
import torchvision
import numpy as np
from torch.nn.functional import batch_norm

from dataloader import *
from img.dataloader import Si_Dataset_extract
from simclr import SimCLR
from simclr.modules import get_resnet
from simclr.modules.transformations import TransformsSimCLR
import pandas as pd
import yaml

def yaml_config_hook(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg

def get_features(loader, model, device):
    feature_vector = []

    for step, (x,j) in enumerate(loader):
        x = x.to(device)

        with torch.no_grad():
            h, _, z, _ = model(x, x)
        h = h.detach()
        feature_vector.extend(h.cpu().detach().numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector

if __name__ == "__main__":
    city = "Shanghai"

    si_path = ""
    df = pd.read_csv(si_path, header=0)
    img_list = df['satellite_img_name'].tolist()
    print(len(img_list))

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("/ssd-data2/nkl2024/satellite-POI/config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    train_dataset = Si_Dataset_extract(
        si_path,
        "",
        transform=TransformsSimCLR(size=args.image_size),
    )

    print('train_set', len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features

    model = SimCLR(encoder, args.projection_dim, n_features)

    model_path = f"model.tar"
    model.load_state_dict(torch.load(model_path, map_location=args.device.type, weights_only=True))
    model = model.to(args.device)
    model.eval()


    feature = get_features(
        train_loader, model,  args.device
    )

    np.save(f"feature.npy", feature)
    print("Features shape {}".format(feature.shape))
