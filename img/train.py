import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import argparse
import os
from dataloader import *
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml

from simclr.modules import get_resnet
from simclr.modules.loss import Infonce_HN
from simclr.modules.transformations import TransformsSimCLR as TransformsSimCLR_suc  # augmentation: generate positive image pairs
from simclr.modules.sync_batchnorm import convert_model
from model import *

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
def normalize(tensor, dim=1, eps=1e-8):
    norm = tensor.norm(p=2, dim=dim, keepdim=True)
    return tensor / (norm + eps)


def compute_height_stats(height_map):
    valid_mask = ~torch.isnan(height_map)
    coverage = valid_mask.float().mean().item()

    if coverage == 0:
        return torch.tensor([0., 0., 0., 0.], device=height_map.device)

    valid_vals = height_map[valid_mask]
    mean_h = valid_vals.mean()
    max_h = valid_vals.max()

    gx = torch.abs(height_map[:, 1:] - height_map[:, :-1])
    gy = torch.abs(height_map[1:, :] - height_map[:-1, :])
    rough = (gx.nan_to_num().mean() + gy.nan_to_num().mean()) / 2

    return torch.tensor([mean_h, max_h, rough, coverage], device=height_map.device)


class MemoryBank:
    def __init__(self, size, feat_dim):
        self.size = size
        self.ptr = 0

        self.feats = torch.zeros(size, feat_dim).cuda()
        self.height_stats = torch.zeros(size, 4).cuda()
        self.filled = 0

    @torch.no_grad()
    def update(self, new_feats, new_height_raw):
        new_feats = F.normalize(new_feats, dim=1)

        B = new_feats.size(0)

        batch_stats = []
        for i in range(B):
            stats = compute_height_stats(new_height_raw[i])
            batch_stats.append(stats)

        batch_stats = torch.stack(batch_stats, dim=0)

        end = self.ptr + B
        if end <= self.size:
            self.feats[self.ptr:end] = new_feats
            self.height_stats[self.ptr:end] = batch_stats
            self.ptr = end
        else:
            first = self.size - self.ptr
            self.feats[self.ptr:] = new_feats[:first]
            self.height_stats[self.ptr:] = batch_stats[:first]

            remain = B - first
            if remain > 0:
                self.feats[:remain] = new_feats[first:]
                self.height_stats[:remain] = batch_stats[first:]
            self.ptr = remain

        self.filled = min(self.filled + B, self.size)

    @torch.no_grad()
    def get_hard_negatives(self, anchor_feat, anchor_height_raw,
                           K_visual=50, topk=10):
        B, D = anchor_feat.shape
        bank_feats = self.feats[:self.filled]
        bank_heights = self.height_stats[:self.filled]

        anchor_norm = F.normalize(anchor_feat, dim=1)
        bank_norm = bank_feats
        sim = anchor_norm @ bank_norm.T

        K_visual = min(K_visual, self.filled)
        _, idx_visual = torch.topk(sim, K_visual, dim=1)

        anchor_stats = []
        for i in range(B):
            anchor_stats.append(compute_height_stats(anchor_height_raw[i]))
        anchor_stats = torch.stack(anchor_stats, dim=0)  #

        cand_stats = bank_heights[idx_visual]

        anchor_stats_exp = anchor_stats.unsqueeze(1).expand(-1, K_visual, -1)

        diff = torch.abs(anchor_stats_exp - cand_stats)
        diff_score = (diff * torch.tensor([1., 1., 0.5, 2.]).cuda()).sum(dim=-1)

        topk = min(topk, K_visual)
        _, idx_h = torch.topk(diff_score, topk, dim=1)

        final_idx = idx_visual.gather(1, idx_h)
        hard_neg_feats = bank_feats[final_idx]

        return hard_neg_feats


def train(args, train_loader, model, criterion, optimizer, memory_bank):
    loss_epoch = 0

    for step, (x_i, x_j, height_i) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)
        height_i = height_i.cuda(non_blocking=True)

        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j, height_i, memory_bank)

        loss.backward()

        optimizer.step()

        memory_bank.update(z_i.detach(), height_i.detach())

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def main(gpu, args, city):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.cuda.set_device(gpu)
    args.device = torch.device(f'cuda:{gpu}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dir = f"data/{city}_example"
    train_dataset = Si_Dataset(
        f"{data_dir}/{city}_si_list.csv",   # region list (satellite_img_name column)
        f"{data_dir}/nearest.txt",          # positive indices, generated by construct_positive.py
        f"{data_dir}/satellite_imagery",    # satellite image directory
        f"{data_dir}/height_crop_raw.npy",  # 64x64 height crops from height.py; stats are computed on the fly
        transform=TransformsSimCLR_suc(size=args.image_size),
    )


    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )
    print('train_dataset', len(train_dataset))

    memory_bank = MemoryBank(
        size=len(train_dataset),
        feat_dim=args.projection_dim
    )

    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features


    model = SimCLR(encoder, args.projection_dim, n_features)
    model = model.to(args.device)

    optimizer, scheduler = load_optimizer(args, model)
    criterion = Infonce_HN(args.temperature)

    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if epoch < 160:
            lr_value = 1e-3
        elif epoch < 220:
            lr_value = 8e-4
        else:
            lr_value = 3e-4

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_value

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        loss_epoch = train(args, train_loader, model, criterion, optimizer, memory_bank)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 50 == 0:
            save_model(args, model, optimizer, city)

        if args.nr == 0:
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr_value, 5)}"
            )
            args.current_epoch += 1

    save_model(args, model, optimizer, city)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    parser.add_argument("--city", type=str, default="Shenzhen", help="city name")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        help="path to the config yaml file",
    )
    config = yaml_config_hook(parser.parse_known_args()[0].config)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    print(args.city)
    main(args.gpu, args, args.city)

