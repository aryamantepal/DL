import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import glob

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),                   
    transforms.Normalize([0.5]*3, [0.5]*3)     
])

class PokemonSpriteDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(image_dir, '*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

sprite_dir = "/common/GAN/archive"  
dataset = PokemonSpriteDataset(sprite_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
