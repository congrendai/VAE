import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os
from model import VAE
from dataset import Camelyonpatch
from torchvision import transforms
from torch.utils.data import DataLoader


image_size = 64
batch_size = 32
latent_dims = 256
epochs = 3000
save_interval = 100
lr = 1e-4
cuda = 1

image_dir = '/media/ssd/congren/camelyonpatch/d02_r0p938c15/train/T_A_64'
save_dir = "/media/ssd/congren/outputs"
log_path = f"{save_dir}/loss_log.csv"

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Resize as needed
    transforms.ToTensor(),
])

# Dataloader
train = Camelyonpatch(image_dir=image_dir, transform=transform)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
model = VAE(in_channels=3, latent_dims=latent_dims).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')  # or 'mean'
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl_loss

# Output directory
os.makedirs(save_dir, exist_ok=True)

# Training function
def train_vae(model, dataloader, epochs):
    model.train()
    for epoch in range(epochs):
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_loss = 0.0
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(images)
            recon_loss, kl_loss = vae_loss(recon, images, mu, logvar)
            
            loss = recon_loss + kl_loss
            loss.backward()

            optimizer.step()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(dataloader.dataset)
        avg_recon_loss = total_recon_loss / len(dataloader.dataset)
        avg_kl_loss = total_kl_loss / len(dataloader.dataset)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{epochs}], Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, Total Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        log_file.write(f"{epoch+1}, {avg_recon_loss:.4f}, {avg_kl_loss:.4f}, {avg_loss:.4f}, {current_lr:.6f}\n")
        log_file.flush()

        # Save reconstructed images
        with torch.no_grad():
            if (epoch + 1) % save_interval == 0:
                # Save the first 8 images and their reconstructions
                sample = images[:8]
                recon, _, _ = model(sample)
                comparison = torch.cat([sample, recon], dim=0)
                z = torch.randn(8, latent_dims).to(device)
                recon_z = model.decode(z)
                comparison = torch.cat([comparison, recon_z], dim=0)
                save_image(comparison, f"{save_dir}/recon_epoch_{epoch+1}.png", nrow=8, normalize=True)

                # Save the model state
                model_path = f"{save_dir}/vae_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path}")


log_file = open(log_path, "w")
log_file.write(f"Using Device: {device}, Batch Size: {batch_size}, Latent Dims: {latent_dims}, Learning Rate: {lr}\n")
log_file.write("epoch,recon_loss,kl_loss,total_loss,learning_rate\n")
train_vae(model, train_loader, epochs=epochs)
log_file.close()