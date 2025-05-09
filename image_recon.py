import torch
from torchvision.utils import save_image
import os
from model import VAE
from dataset import Camelyonpatch
from torchvision import transforms
from torch.utils.data import DataLoader

# Parameters
cuda = 1
latent_dims = 256
in_channels = 3
image_size = 64
device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

# Settings
batch_size = 32
output_dir = "/media/ssd/congren/recon"
val_dir = '/media/ssd/congren/camelyonpatch/d02_r0p938c15/valid/T_64_ddpm'
weight_dir = "/media/ssd/congren/outputs/vae_epoch_1500.pth"
os.makedirs(output_dir, exist_ok=True)

# Load model
vae = VAE(in_channels=in_channels, latent_dims=latent_dims).to(device)
vae.load_state_dict(torch.load(weight_dir, map_location=device))
vae.eval()

val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Resize as needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Normalize to [-1, 1]
])

# Dataloader for validation set
val_dataset = Camelyonpatch(image_dir=val_dir, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Reconstruct and save images
for i, (images, _) in enumerate(val_loader):
    images = images.to(device)
    with torch.no_grad():
        recon_images, _, _ = vae(images)

    for j in range(recon_images.size(0)):
        save_path = os.path.join(output_dir, f"img_{i * batch_size + j:06d}.png")
        save_image(recon_images[j], save_path, normalize=True)

print("Image reconstruction completed.")
