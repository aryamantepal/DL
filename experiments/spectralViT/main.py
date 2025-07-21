import torch
from torch.nn import nn
from torch.utils.data import DataLoader, random_split
from dataloader.indianPines import IndianPinesDataset
from model.model import SpectralViT

config = {
    "data_path": "/Users/aryamantepal/Desktop/Tufts2024/data/Indian_pines_corrected.mat",
    "gt_path": "/Users/aryamantepal/Desktop/Tufts2024/data/Indian_pines_gt.mat",
    "patch_size": 5,
    "img_size": (5, 5),
    "batch_size": 64,
    "emb_dim": 64,
    "depth": 4,
    "num_heads": 4,
    "mlp_dim": 128,
    "num_classes": 16,
    "lr": 1e-3,
    "epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

dataset = IndianPinesDataset(config["data_path"], config["gt_path"], config["patch_size"])
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

model = SpectralViT(
    in_channels=dataset.data.shape[-1],
    img_size=config["img_size"],
    patch_size=1,  
    emb_dim=config["emb_dim"],
    depth=config["depth"],
    num_heads=config["num_heads"],
    mlp_dim=config["mlp_dim"],
    num_classes=config["num_classes"]
).to(config["device"])

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
criterion = nn.CrossEntropyLoss()

for epoch in range(config["epochs"]):
    model.train()
    for x, y in train_loader:
        x, y = x.to(config["device"]), y.to(config["device"])
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        pass
    
    print(f"Epoch {epoch+1}/{config['epochs']}")