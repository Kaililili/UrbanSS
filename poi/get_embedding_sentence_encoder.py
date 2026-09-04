import argparse
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--city", type=str, default="Shanghai")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
city = args.city

csv_path = f""
model_path = ""
output_path = f""
task = "cat"  #  cat / surrounding


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )


def extract_info(prompt, info_type):
    if info_type == "basic":
        pattern = r"Basic Information:(.*?)\."
    elif info_type == "category":
        pattern = r"category Information:(.*?)\."
    elif info_type == "surrounding":
        pattern = r"Surrounding Information:(.*?)\."
    else:
        return ""

    match = re.search(pattern, prompt)
    return match.group(1).strip() if match else ""


print(city)
print(task)

poi_df = pd.read_csv(csv_path)
print(f"Total POIs in {city}: {len(poi_df)}")

new_inputs = []
for _, row in poi_df.iterrows():
    s_prompt = row["surrounding_prompt"]
    c_prompt = row["cat_prompt"]

    basic_info = extract_info(s_prompt, "basic")
    category_info = extract_info(s_prompt, "category")
    surrounding_info = extract_info(c_prompt, "surrounding")

    if task == "cat":
        new_prompt = f"Basic Information: {basic_info}.\nCategory Information: {category_info}."
    else:
        new_prompt = f"Basic Information: {basic_info}.\nSurrounding Information: {surrounding_info}."
    new_inputs.append(new_prompt.strip())

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)
model.eval()

tokenized = tokenizer(new_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
input_ids = tokenized["input_ids"].to(device)
attention_mask = tokenized["attention_mask"].to(device)

batch_size = 16
dataloader = DataLoader(range(len(new_inputs)), batch_size=batch_size, shuffle=False)

embeddings = torch.zeros(
    len(new_inputs), model.config.hidden_size, dtype=torch.float32, device="cpu"
)

offset = 0
with torch.no_grad():
    for batch_idx in tqdm(dataloader, desc=f"Encoding {city} sentences"):
        batch_input_ids = input_ids[batch_idx].to(device)
        batch_attention_mask = attention_mask[batch_idx].to(device)

        outputs = model(batch_input_ids, attention_mask=batch_attention_mask, return_dict=True)
        sentence_embeddings = mean_pooling(outputs, batch_attention_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        embeddings[offset:offset + batch_input_ids.shape[0]] = sentence_embeddings.cpu()
        offset += batch_input_ids.shape[0]

embeddings_np = embeddings.numpy()
print(f"Embeddings shape: {embeddings_np.shape}")

np.save(output_path, embeddings_np)
print(f"Embeddings saved to: {output_path}")
