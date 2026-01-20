import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from info_nce import InfoNCE, info_nce
from tqdm import tqdm
import random, numpy as np, torch, os
from model import *

def extract_embed(model, poi_dataloader, save_path, device):
    model = model.to(device)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    all_embeds = []

    with torch.no_grad():
        for batch in tqdm(poi_dataloader, desc="Extracting embeddings"):
            batch = batch.to(device)
            out = model(batch)
            out = out.squeeze(1)
            all_embeds.append(out.cpu())

    all_embeds = torch.cat(all_embeds, dim=0).numpy()
    print(all_embeds.shape)
    np.save(f"", all_embeds)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed = 1
    set_seed(seed)

    LR = 5e-3
    BATCH_SIZE = 128
    EPOCH = 500
    SAVE_INTERVAL = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    city = "Shanghai"

    llmembed_path1 = f""
    llmembed_path2 = f""
    contra_path = f""  #constrat data. region with the similar POI distributions

    dataset = POIContrastDataset(contra_path, device)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PoiSemantic(llmembed_path1, llmembed_path2, device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    nceloss = InfoNCE(temperature=0.1, reduction='mean', negative_mode='paired')

    best_loss = float("inf")
    best_state = None

    patience = 40
    patience_counter = 0

    for epoch in range(EPOCH):
        model.train()
        l = []
        for batch in tqdm(train_dataloader):
            batch = batch.to(device)
            z = model(batch)

            query, positive, negative = z[:, 0, :], z[:, 1, :], z[:, 2:, :]

            query_ = query.squeeze(1)
            positive_ = positive.squeeze(1)

            z = rearrange(z, 'b n d -> (b n) d')

            loss = nceloss(query_, positive_, negative)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            l.append(loss.item())

        avg_loss = sum(l) / len(l)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(device)
                z = model(batch)
                q, p, n = z[:, 0, :], z[:, 1, :], z[:, 1:, :]
                q_, p_ = q.squeeze(1), p.squeeze(1)
                loss_val = nceloss(q_, p_, n)
                val_losses.append(loss_val.item())
        val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}")

        save_path = f""

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            torch.save(best_state, save_path)
            print(f"✅ Saved new best model at epoch {epoch + 1}, val_loss={val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    print(f"Training done. Best val_loss={best_loss:.4f}")

    #########save embed ################
    model.eval()
    region_num = 4584
    poi_dataset = PoiExtractDataset(city, region_num, device)
    poi_dataloader = DataLoader(poi_dataset, batch_size=128, shuffle=False)
    extract_embed(model, poi_dataloader, save_path, device)

