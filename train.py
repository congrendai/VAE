import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os
from model import VAE
from dataset import Camelyonpatch
from torchvision import transforms
from torch.utils.data import DataLoader

from pytorch_msssim import ssim
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor

image_size = 64
batch_size = 32
latent_dims = 128
epochs = 3000
save_interval = 100
lr = 1e-4
cuda = 0

# KL annealing parameters
beta = 1
beta_step = 1000

# Paths
image_dir = '/media/ssd/congren/camelyonpatch/d02_r0p938c15/train/T_A_64'
save_dir = "/media/ssd/congren/outputs_all"
log_path = f"{save_dir}/loss_log.csv"

# Transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # [0,1] → [-1,1]
])

# Dataloader
train = Camelyonpatch(image_dir=image_dir, transform=transform)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
# Model
model = VAE(in_channels=3, latent_dims=latent_dims).to(device)
# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Loss functions
vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval().to(device)
for p in vgg.parameters():
    p.requires_grad = False

def perceptual_loss(x, y):
    fx = vgg(x)
    fy = vgg(y)
    return F.mse_loss(fx, fy)

def ssim_loss(x, y):
    return 1 - ssim(x, y, data_range=1.0, size_average=True)

def mse_loss(x, y):
    return F.mse_loss(x, y, reduction='mean')

def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # Also mean

def vae_loss(recon_x, x, mu, logvar, beta):
    mse = mse_loss(recon_x, x)
    perceptual = perceptual_loss(recon_x, x)
    ssim = ssim_loss(recon_x, x)
    recon_loss = mse + perceptual + ssim
    kl = kl_loss(mu, logvar)
    return recon_loss, beta * kl

# Output directory
os.makedirs(save_dir, exist_ok=True)

# Training function
def train_vae(model, dataloader, epochs):
    model.train()
    for epoch in range(epochs):
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_loss = 0.0

        # Linear KL annealing schedule: β increases from 0 to 1 over 1000 epochs
        # beta = min(1.0, epoch / beta_step)

        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(images)
            recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, beta)

            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{epochs}], β: {beta:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, Total Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        log_file.write(f"{epoch+1},{avg_recon_loss:.4f},{avg_kl_loss:.4f},{avg_loss:.4f},{current_lr:.6f}\n")
        log_file.flush()

        if (epoch + 1) % save_interval == 0:
            with torch.no_grad():
                sample = images[:8]
                recon, _, _ = model(sample)
                comparison = torch.cat([sample, recon], dim=0)
                z = torch.randn(8, latent_dims).to(device)
                recon_z = model.decode(z)
                comparison = torch.cat([comparison, recon_z], dim=0)
                save_image(comparison, f"{save_dir}/recon_epoch_{epoch+1}.png", nrow=8, normalize=True)

                model_path = f"{save_dir}/vae_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path}")

log_file = open(log_path, "w")
log_file.write(f"Using Device: {device}, Batch Size: {batch_size}, Latent Dims: {latent_dims}, Learning Rate: {lr}\n")
log_file.write("epoch,recon_loss,kl_loss,total_loss,learning_rate\n")
train_vae(model, train_loader, epochs=epochs)
log_file.close()