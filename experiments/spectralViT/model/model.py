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
        x = self.proj(x)  
        x = x.flatten(2).transpose(1, 2)  
        return x + self.pos_embed

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


