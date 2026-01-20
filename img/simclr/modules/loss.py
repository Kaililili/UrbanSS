import torch
import torch.nn as nn
import torch.nn.functional as F


class Infonce_HN(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z_i, z_j, height_raw, memory_bank,
                K_visual=50, topk=10):
        B, D = z_i.shape
        device = z_i.device

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        z_all = torch.cat([z_i, z_j], dim=0)  # (2B, D)

        sim_matrix = (z_all @ z_all.t()) / self.temperature

        pos_indices = torch.arange(B, device=device)
        positives = torch.cat([
            sim_matrix[pos_indices, pos_indices + B],
            sim_matrix[pos_indices + B, pos_indices]
        ], dim=0).view(2 * B, 1)   # (2B,1)

        mask = torch.ones_like(sim_matrix, dtype=bool)
        mask.fill_diagonal_(False)

        mask[pos_indices, pos_indices + B] = False
        mask[pos_indices + B, pos_indices] = False

        negatives_batch = sim_matrix[mask].view(2 * B, -1)

        neg_hard = memory_bank.get_hard_negatives(
            anchor_feat=z_i,
            anchor_height_raw=height_raw,
            K_visual=K_visual,
            topk=topk
        )

        sim_hard = torch.einsum('bd,bkd->bk', z_i, neg_hard) / self.temperature

        sim_hard_2 = torch.cat([sim_hard, sim_hard], dim=0)

        negatives = torch.cat([negatives_batch, sim_hard_2], dim=1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * B, dtype=torch.long, device=device)

        loss = self.criterion(logits, labels)
        return loss
