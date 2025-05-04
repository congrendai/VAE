import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Camelyonpatch(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(image_dir) if f.endswith('.png')
        ]
        self.image_files.sort()  # Optional: ensure consistent order

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label = int(img_name.split('_')[0])
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
