import torch
from torchvision.utils import save_image
import os
from model import VAE

# --- Model definitions ---
# (Ensure your full VAE, Decoder class definitions are included above or imported)

# Parameters (set to your actual training setup)
latent_dims = 128  # Change if different in your setup
in_channels = 3    # RGB
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained VAE
vae = VAE(in_channels=in_channels, latent_dims=latent_dims).to(device)
vae.load_state_dict(torch.load("/home/gyang/Desktop/VAE/outputs/vae_epoch_1800.pth", map_location=device))
vae.eval()

# Use decoder
decoder = vae.decoder

# Settings for generation
num_images = 32768
batch_size = 1
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Generate and save
with torch.no_grad():
    for i in range(0, num_images, batch_size):
        z = torch.randn(batch_size, latent_dims).to(device)
        imgs = decoder(z)  # Output: (batch_size, 3, 64, 64)

        for j, img in enumerate(imgs):
            save_path = os.path.join(output_dir, f"img_{i + j:06d}.png")
            save_image(img, save_path)

print("Image generation completed.")
