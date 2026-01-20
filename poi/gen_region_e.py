import pandas as pd
import numpy as np
import random

city = "Shanghai"
region_path = f""
region_df = pd.read_csv(region_path)

region_ids = list(range(len(region_df)))
region_ids = [str(rid) for rid in region_ids]
name2id = {name: idx for idx, name in enumerate(region_df["satellite_img_name"].unique())}

poi_df = pd.read_csv(f"")

poi_embeds = np.load(f"")
region2embeds = {}
for i, row in poi_df.iterrows():
    rid_val = row["region_id"]
    if pd.isna(rid_val):
        continue
    rid = str(int(rid_val))
    if rid not in region2embeds:
        region2embeds[rid] = []
    region2embeds[rid].append(poi_embeds[i])

final_embeddings = []
for rid in region_ids:
    if rid in region2embeds:
        embs = np.stack(region2embeds[rid], axis=0)  # [num_poi, dim]
        final_emb = embs.mean(axis=0)

    final_embeddings.append(final_emb)

final_embeddings = np.stack(final_embeddings, axis=0)  # [num_region, dim]

np.save(f"", final_embeddings)
