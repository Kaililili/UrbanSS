import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import random
import numpy as np
from transformers import logging
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Extract POI semantic embeddings")
parser.add_argument("--city", type=str, default="Shanghai", help="city name")
parser.add_argument("--gpu", type=int, default=0, help="GPU index")
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("")
model = AutoModelForCausalLM.from_pretrained("").half().to(device)



city = args.city
#get prompt
poi_df = pd.read_csv(f'')
task = "surrounding"
#task = "cat"

print(city)
print(task)

prompts = poi_df['surrounding_prompt'].tolist()
print(len(prompts))

tokenized = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
input_ids = tokenized['input_ids'].to(device)
attention_mask = tokenized['attention_mask'].to(device)

bs = 8  # batch size
dataloader = DataLoader(range(len(prompts)), batch_size=bs, shuffle=False)


last_layer_activations = torch.zeros(len(prompts), model.config.hidden_size, dtype=torch.float16, device="cpu")

offset = 0
with torch.no_grad():
    for batch_idx in tqdm(dataloader, desc="Extracting embeddings"):
        batch_input_ids = input_ids[batch_idx].to(device)
        batch_attention_mask = attention_mask[batch_idx].to(device)

        out = model(batch_input_ids, attention_mask=batch_attention_mask,
                    output_hidden_states=True, return_dict=True)


        activation = out.hidden_states[-1]  # [batch, seq_len, hidden_dim]

        last_valid_ix = batch_attention_mask.sum(dim=1) - 1
        processed = activation[torch.arange(batch_input_ids.shape[0], device=device), last_valid_ix, :]

        last_layer_activations[offset:offset+batch_input_ids.shape[0]] = processed.cpu().to(torch.float16)

        offset += batch_input_ids.shape[0]


last_layer_embedding = last_layer_activations.numpy()

np.save(f"", last_layer_embedding)
