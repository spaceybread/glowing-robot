import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, latent_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embed.weight, mean=0, std=1.0)
        self.net = nn.Sequential(
            nn.Conv3d(embed_dim, 32, kernel_size=4, stride=2, padding=1),  # 16x16x64
            nn.ReLU(), nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),         # 8x8x32
            nn.ReLU(), nn.BatchNorm3d(64),
            nn.Conv3d(64, latent_dim, kernel_size=4, stride=2, padding=1), # 4x4x16
            nn.Tanh(),
        )

    def forward(self, x):
        # x: (B, 32, 32, 128) integer block ids
        x = self.embed(x)                    # (B, 32, 32, 128, embed_dim)
        x = x.permute(0, 4, 1, 2, 3)        # (B, embed_dim, 32, 32, 128)
        return self.net(x)                   # (B, latent_dim, 2, 2, 8)

class Decoder(nn.Module):
    def __init__(self, vocab_size, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(), nn.Dropout3d(0.1),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(), nn.Dropout3d(0.1),
            nn.ConvTranspose3d(32, vocab_size, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        return self.net(z)  # (B, vocab_size, 32, 32, 128) — logits per block type

class Autoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, latent_dim)
        self.decoder = Decoder(vocab_size, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
