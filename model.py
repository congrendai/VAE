import torch
import torch.nn.functional as F
from torch import nn

def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

def conv_transpose_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=0, with_act=True):
    modules = [
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        ),
    ]
    if with_act: # Controlling this will be handy later
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU())
    return nn.Sequential(*modules)

class VAEEncoder(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super().__init__()

        self.conv_layers = nn.Sequential(
            conv_block(in_channels, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 1024),
        )
        # Define fully connected layers for mean and log-variance
        self.mu = nn.Linear(1024 * 4 * 4, latent_dims)
        self.logvar = nn.Linear(1024 * 4 * 4, latent_dims)
    
    def forward(self, x):
        # breakpoint()
        bs = x.shape[0]
        x = self.conv_layers(x)
        x = x.reshape(bs, -1)  # Flatten the output
        mu = self.mu(x)
        logvar = self.logvar(x)
        return (mu, logvar)

class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dims):
        super().__init__()
        self.linear = nn.Linear(latent_dims, 1024 * 4 * 4)
        self.t_conv_layers = nn.Sequential(
            conv_transpose_block(1024, 512, kernel_size=4, stride=2, padding=1),      # 4x4 → 8x8
            conv_transpose_block(512, 256, kernel_size=4, stride=2, padding=1),       # 8x8 → 16x16
            conv_transpose_block(256, 128, kernel_size=4, stride=2, padding=1),       # 16x16 → 32x32
            conv_transpose_block(128, out_channels, kernel_size=4, stride=2, padding=1, with_act=False)  # 32x32 → 64x64
        )
        # self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

    def forward(self, x):
        bs = x.shape[0]
        # Reshape the latent vector to match the input shape of the transposed conv layers
        x = self.linear(x)
        x = x.reshape((bs, 1024, 4, 4))
        # Pass through the transposed conv layers
        x = self.t_conv_layers(x)
        # Apply sigmoid activation to get pixel values in range [0, 1]
        # x = self.sigmoid(x)
        x = self.Tanh(x)
        return x

class VAE(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dims)
        self.decoder = Decoder(in_channels, latent_dims)

    def encode(self, x):
        # Returns mu, log_var
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        # Obtain parameters of the normal (Gaussian) distribution
        mu, logvar = self.encode(x)
        # Sample from the distribution
        std = torch.exp(0.5 * logvar)
        z = self.sample(mu, std)
        # Decode the latent point to pixel space
        reconstructed = self.decode(z)
        # Return the reconstructed image, and also the mu and logvar
        # so we can compute a distribution loss
        return reconstructed, mu, logvar

    def sample(self, mu, std):
    # Reparametrization trick
    # Sample from N(0, I), translate and scale
        eps = torch.randn_like(std)
        return mu + eps * std