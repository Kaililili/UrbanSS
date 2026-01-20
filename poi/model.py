import numpy as np
import pandas as pd
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader

class WeightedSemanticFusion(nn.Module):
    def __init__(self, dim, dim_fused, device):
        super().__init__()
        self.W = nn.Linear(dim, dim_fused, bias=False).to(device)
        self.f = nn.Linear(dim_fused * 2, 1).to(device)
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        f = self.W(x)
        f1, f2 = f[0], f[1]
        a12 = self.act(self.f(torch.cat([f1, f2], dim=-1)))
        a21 = self.act(self.f(torch.cat([f2, f1], dim=-1)))

        a1 = torch.mean(a12, dim=0)
        a2 = torch.mean(a21, dim=0)

        coef = torch.cat([a1, a2], dim=-1)
        coef = F.softmax(coef, dim=-1)

        return coef


class Embed2hidden(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.to_hidden = nn.Linear(dim, hidden_dim, bias=False).float()

    def forward(self, x):
        hidden = self.to_hidden(x)
        return F.normalize(hidden, dim=-1)


class Embed_layer(
    nn.Module):
    def __init__(self, device, hidden_dim=1024, dim_reduct=True, embed_path=''):
        super().__init__()
        self.embed = torch.from_numpy(np.load(embed_path)).float().to(device)
        num, dim = self.embed.shape
        self.shape = self.embed.shape

        self.embedding_layer = nn.Embedding(num, dim, _weight=self.embed)

        self.dim_reduct = dim_reduct
        if self.dim_reduct == True:
            self.embed2hidden = Embed2hidden(dim, hidden_dim).to(device)
        else:
            self.embed2hidden = nn.Identity()

        for p in self.embedding_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.to(torch.float32)
        out = self.embed2hidden(x)
        return out

    def get_shape(self):
        return self.shape

class PoiSemantic(nn.Module):
    def __init__(self,  llm_e_path1, llm_e_path2, device,dim=1024):

        super().__init__()

        self.intrin_layer1 = Embed_layer(embed_path = llm_e_path1, hidden_dim=dim, device=device)
        self.context_layer2 = Embed_layer(embed_path = llm_e_path2, hidden_dim=dim, device=device)

        self.semantic_weight=  WeightedSemanticFusion(dim=dim, dim_fused= 2 * 256, device=device)

    def forward(self, batch):

        intrin_embed = self.intrin_layer1(batch)

        context_embed = self.context_layer2(batch)

        if intrin_embed.dim() == 2:
            intrin_embed = intrin_embed.unsqueeze(1)  # [B, 1, D]
        if context_embed.dim() == 2:
            context_embed = context_embed.unsqueeze(1)  # [B, 1, D]

        semantic = torch.stack([intrin_embed, context_embed])

        w = rearrange(semantic, 'fn b n d -> fn (b n) d')

        weight = self.semantic_weight(w)

        z = weight[0] * semantic[0] + weight[1] * semantic[1]
        return z


class POIContrastDataset(data.Dataset):
    def __init__(self, path, device, simple=False):

        df = pd.read_csv(path, sep=',', header=0, dtype={'anchor': int, 'positive': int,
                                                         'negative': str})
        if simple == 'True':
            df = df.sample(frac=0.000001)
        df['negative'] = df['negative'].apply(lambda x: eval(x))
        self.device = device
        self.data = df
        self.data = self.data.values

    def __getitem__(self, index):
        anchor, pos, negative = self.data[index]
        data = [anchor, pos] + negative
        data = torch.IntTensor(data).to(self.device)

        return data

    def __len__(self):
        return len(self.data)

class PoiExtractDataset(data.Dataset):
    def __init__(self, city, region_num, device):
        self.region_num = region_num

        self.device = device

    def __getitem__(self, index):
        data = torch.tensor(index, dtype=torch.long, device=self.device)
        return data

    def __len__(self):
        return self.region_num
