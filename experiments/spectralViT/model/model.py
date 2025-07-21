import torch
import torch.nn as nn

class SpectralPatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, img_size):
        super().__init__()
        self.H, self.W = img_size
        self.patch_size = patch_size
        self.n_patches = (self.H // patch_size) * (self.W // patch_size)
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, emb_dim))
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, emb_dim, H//p, W//p)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, emb_dim)
        x = x + self.pos_embed
        return x

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x



class SpectralViT(nn.Module):
    def __init__(self, *, in_channels, img_size, patch_size, emb_dim, depth, num_heads, mlp_dim, num_outputs, use_cls_token=False):
        super().__init__()
        self.patch_embed = SpectralPatchEmbedding(in_channels, patch_size, emb_dim, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim)) if use_cls_token else None
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(emb_dim, num_heads, mlp_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_outputs)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, N, D)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        x = x[:, 0] if self.cls_token is not None else x.mean(dim=1)
        return self.head(x)


indian_pines_path = "/Users/aryamantepal/Desktop/Tufts2024/data/Indian_pines_corrected.mat"
indian_pines_path_gt = "/Users/aryamantepal/Desktop/Tufts2024/data/Indian_pines_gt.mat"

from scipy.io import loadmat
import numpy as np


data = loadmat(indian_pines_path)["indian_pines_corrected"]  # shape: (145, 145, 220)
labels = loadmat(indian_pines_path_gt)["indian_pines_gt"]     # shape: (145, 145)

def normalize_hsi(data):
    data = data.astype(np.float32)
    for i in range(data.shape[-1]):
        band = data[:, :, i]
        band -= band.min()
        band /= band.max()
        data[:, :, i] = band
    return data

data = normalize_hsi(data)  # shape: (145, 145, bands)

def extract_patches(data, labels, patch_size=5):
    pad = patch_size // 2
    h, w, c = data.shape
    data_padded = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    patches, targets = [], []

    for i in range(pad, h + pad):
        for j in range(pad, w + pad):
            label = labels[i - pad, j - pad]
            if label == 0:
                continue  # skip unlabeled
            patch = data_padded[i - pad:i + pad + 1, j - pad:j + pad + 1, :]
            patches.append(patch)
            targets.append(label - 1)  # labels from 0 to 15

    return np.array(patches), np.array(targets)

patches, targets = extract_patches(data, labels, patch_size=5)
print(patches.shape)  # (N, 5, 5, bands)
print(targets.shape)  # (N,)

import torch
from torch.utils.data import Dataset, DataLoader

class IndianPinesPatches(Dataset):
    def __init__(self, patches, labels):
        self.patches = torch.from_numpy(patches).permute(0, 3, 1, 2).float()  # (N, C, H, W)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]

dataset = IndianPinesPatches(patches, targets)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SpectralViT(
    in_channels=data.shape[-1],  
    img_size=(5, 5),             
    patch_size=1,                
    emb_dim=64,
    depth=4,
    num_heads=4,
    mlp_dim=128,
    num_outputs=16,              
    use_cls_token=True,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

model.train()
for epoch in range(10):
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
