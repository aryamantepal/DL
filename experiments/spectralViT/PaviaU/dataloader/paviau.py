import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

class PaviaUDataset(Dataset):
    def __init__(self, data_path, gt_path, patch_size=5, normalize=True):
        self.patch_size = patch_size
        self.data = loadmat(data_path)["PaviaU"]
        self.labels = loadmat(gt_path)["PaviaU_gt"]
        if normalize:
            self.data = self.normalize_hsi(self.data)
        self.patches, self.targets = self.extract_patches()
        
    def normalize_hsi(self, data):
        data = data.astype(np.float32)
        for i in range(data.shape[-1]):
            band = data[:, :, i]
            mean = band.mean()
            std = band.std()
            data[:, :, i] = (band - mean) / (std + 1e-6)
        return data
    
    def extract_patches(self):
        pad = self.patch_size // 2
        h, w, c = self.data.shape
        data_padded = np.pad(self.data, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        patches, targets = [], []

        for i in range(pad, h + pad):
            for j in range(pad, w + pad):
                label = self.labels[i - pad, j - pad]
                if label == 0:
                    continue
                patch = data_padded[i - pad:i + pad + 1, j - pad:j + pad + 1, :]
                patches.append(patch)
                targets.append(label - 1)

        return np.array(patches), np.array(targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        patch = torch.from_numpy(self.patches[idx]).float().permute(2, 0, 1)
        label = torch.tensor(self.targets[idx]).long()
        return patch, label