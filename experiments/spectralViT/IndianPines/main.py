import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dataloader.indianPines import IndianPinesDataset
from model.model import SpectralViT
import argparse

            
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
args = parser.parse_args()

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
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": args.epochs,
}

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    preds, truths = [], []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        truths.extend(y.cpu().numpy())
    
    acc = accuracy_score(truths, preds)
    return running_loss/len(loader), acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds, truths = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            
            running_loss += loss.item()
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            truths.extend(y.cpu().numpy())
    
    acc = accuracy_score(truths, preds)
    cm = confusion_matrix(truths, preds)
    report = classification_report(truths, preds)
    return running_loss/len(loader), acc, cm, report

def main():
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
        num_outputs=config["num_classes"]
    ).to(config["device"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Starting training on {config['device']}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config["device"])
        val_loss, val_acc, cm, report = validate(model, val_loader, criterion, config["device"])
        
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        
        if epoch == config["epochs"] - 1:  
            print("\nConfusion Matrix:")
            print(cm)
            print("\nClassification Report:")
            print(report)

if __name__ == "__main__":
    main()