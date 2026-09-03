import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

class Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.from_numpy(y).float().view(-1, 1)  # float32, shape (N,1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        f1, f2, f3 = self.X[idx]
        label = self.y[idx]
        return (
            torch.tensor(f1, dtype=torch.float32),
            torch.tensor(f2, dtype=torch.float32),
            torch.tensor(f3, dtype=torch.float32),
            label
        )


def train(model, dataloader, criterion, optimizer, device, lambda_ortho=0.1):
    model.train()
    total_loss = 0.0
    for poi, img, height, labels in dataloader:
        poi, img, height, labels = poi.to(device), img.to(device), height.to(device), labels.to(device)

        optimizer.zero_grad()

        loss, _, _ = model([poi, img, height], labels, lambda_ortho)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def test(model, dataloader, criterion, device):

    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []


    with torch.no_grad():
        for poi, img, height, labels in dataloader:
            poi, img, height, labels = poi.to(device), img.to(device), height.to(device), labels.to(device)


            predictions, weight = model([poi, img, height])

            loss = criterion(predictions, labels)
            total_loss += loss.item() * labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.array(all_predictions), np.array(all_labels)

class AdaptiveOSFModule(nn.Module):
    def __init__(self, embedding_dims, common_dim):
        super().__init__()
        self.num_modalities = len(embedding_dims)
        self.common_dim = common_dim


        self.projection_layers = nn.ModuleList([
            nn.Linear(d, common_dim) for d in embedding_dims
        ])


        self.order_net = nn.Sequential(
            nn.Linear(common_dim * self.num_modalities, common_dim),
            nn.ReLU(),
            nn.Linear(common_dim, self.num_modalities)
        )
        self.alpha_net = nn.Linear(common_dim * self.num_modalities, 1)

        self.fuse_layers = nn.ModuleList([
            nn.Linear(2 * common_dim, common_dim)
            for _ in range(self.num_modalities - 1)
        ])

        self.layer_norm = nn.LayerNorm(common_dim)


    def forward(self, embeddings):

        B = embeddings[0].size(0)
        M = self.num_modalities


        projected_embs = [proj(e) for proj, e in zip(self.projection_layers, embeddings)]  # list of (B, D)
        concat_all = torch.cat(projected_embs, dim=-1)  # (B, M*D)


        weights = F.softmax(self.order_net(concat_all), dim=-1)  # (B, M)
        alpha = torch.sigmoid(self.alpha_net(concat_all))  # (B, 1)
        score1 = weights
        score2 = -weights
        final_score = alpha * score1 + (1 - alpha) * score2  # (B, M)
        sorted_idx = torch.argsort(final_score, dim=-1)  # (B, M)

        fused_all = []
        fusion_steps_all = []
        order_list = []

        for b in range(B):
            fused = projected_embs[0][b].new_zeros(self.common_dim)
            steps = []
            order = sorted_idx[b]

            for t in range(M):
                i_idx = order[t].item()
                emb = projected_embs[i_idx][b]
                weight = weights[b, i_idx]

                fused = self.layer_norm(
                    fused + weight * self.fuse_layers[min(t, len(self.fuse_layers) - 1)](
                        torch.cat([fused, emb], dim=-1)
                    )
                )
                fused = F.relu(fused)
                steps.append(fused)

            fused_all.append(fused)
            fusion_steps_all.append(steps)
            order_list.append(order)

        fused_all = torch.stack(fused_all, dim=0)  # (B, D)
        return fused_all, projected_embs, fusion_steps_all, weights, order_list


def orthogonal_loss(fusion_steps_all, projected_embs, order_list, lambda_l1=0.01, lambda_l2=0.01):

    loss = 0.0
    B = len(fusion_steps_all)

    for b in range(B):
        steps = fusion_steps_all[b]
        order = order_list[b]  # (M,)

        for t in range(1, len(steps)):
            F_prev = steps[t - 1]
            emb_idx = order[t].item()
            E_new = projected_embs[emb_idx][b]

            dot = torch.sum(F_prev * E_new)
            loss += lambda_l1 * torch.abs(dot) + lambda_l2 * dot ** 2

    return loss / B


class AdaptiveOSFModel(nn.Module):
    def __init__(self, embedding_dims, common_dim):
        super().__init__()
        self.osf_fusion = AdaptiveOSFModule(embedding_dims, common_dim)
        hidden_dim = common_dim // 2
        self.predictor = nn.Sequential(
            nn.Linear(common_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, embeddings, labels=None, lambda_ortho=0.1):
        fused_embedding, projected_embs, fusion_steps, weights, order_list = self.osf_fusion(embeddings)
        preds = self.predictor(fused_embedding)


        if labels is None:
            return preds, weights, fused_embedding

        loss_task = F.mse_loss(preds, labels)
        loss_ortho = orthogonal_loss(fusion_steps, projected_embs, order_list)
        loss_total = loss_task + lambda_ortho * loss_ortho

        return loss_total, preds, weights


def choose(city, indic):

    feature_path1 = f""
    feature_path2 = f""
    feature_path3 = f""

    feature1 = np.load(feature_path1)
    feature2 = np.load(feature_path2)
    feature3 = np.load(feature_path3)

    region_path = f""

    df_region = pd.read_csv(region_path)
    name2id = {name: idx for idx, name in enumerate(df_region["satellite_img_name"].unique())}

    df_indic = pd.read_csv(f"")
    matched_feature1 = []
    matched_feature2 = []
    matched_feature3 = []
    matched_log_counts = []

    for _, row in df_indic.iterrows():
        name = row["satellite_img_name"]
        if name in name2id:
            idx = name2id[name]
            feat1 = feature1[idx]
            feat2 = feature2[idx]
            feat3 = feature3[idx]
            matched_feature1.append(feat1)
            matched_feature2.append(feat2)
            matched_feature3.append(feat3)

            log = row[f"{indic}_log_transform"]
            matched_log_counts.append(log)
        else:
            print("no name")

    features1 = np.array(matched_feature1)  # (N, D)
    features2 = np.array(matched_feature2)
    features3 = np.array(matched_feature3)

    indicator = np.array(matched_log_counts)

    return features1, features2, features3,  indicator, features1.shape[-1], features2.shape[-1], features3.shape[-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AOSF downstream prediction")
    parser.add_argument("--city", type=str, default="Beijing", help="city name")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    args = parser.parse_args()

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    city = args.city
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    indic = "popcount"
    #indic = "GDP"
    #indic = "CO2"
    #indic = "comment"


    poi, img, height,  indicator, dim_poi, dim_img, dim_height = choose(city, indic)

    X = list(zip(poi, img, height))

    X_train, X_test, y_train, y_test = train_test_split(X, indicator, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2 / 0.8)

    train_dataset = Dataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = Dataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataset = Dataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    dims = [dim_poi, dim_img, dim_height]

    model = AdaptiveOSFModel(dims, common_dim=256).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)


    best_val_loss = float('inf')
    early_stopping_patience = 60
    early_stop_count = 0
    train_losses = []
    val_losses = []

    lambda_ortho = 0.1

    for epoch in range(20000):

        train_loss = train(model, train_loader, criterion, optimizer, device, lambda_ortho)
        train_losses.append(train_loss)


        val_loss, val_preds, val_labels = test(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"")
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}!")
                break

        if epoch % 10 == 0:
            r2_val = r2_score(val_labels, val_preds)
            print(f"Epoch {epoch} - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, R2: {r2_val:.4f}")

    model.load_state_dict(torch.load(f""))

    test_loss, test_pred, test_label = test(model, test_loader, criterion, device)
    rmse = np.sqrt(mean_squared_error(test_label, test_pred))
    r2 = r2_score(test_label, test_pred)
    mae = mean_absolute_error(test_label, test_pred)
    mape = np.mean(np.abs((test_label - test_pred) / test_label))
    print(indic)
    print(f"Test Loss: {test_loss:.4f}, Test R2: {r2:.3f}, Test RMSE: {rmse:.3f}, Test MAE: {mae:.3f}, Test MAPE: {mape:.4f}")


    train_loss, train_pred, train_label = test(model, val_loader, criterion, device)
    r2_train = r2_score(train_label, train_pred)
    print(f"train_loss: {train_loss:.4f} r2_train: {r2_train:.4f}")






