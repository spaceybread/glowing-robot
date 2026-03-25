import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model import Encoder, Decoder

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.decay          = decay

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('codebook',     embed)
        self.register_buffer('cluster_size', torch.ones(num_embeddings))
        self.register_buffer('embed_avg',    embed.clone())

    def forward(self, z):
        B, C, D, H, W = z.shape
        z_flat = z.permute(0,2,3,4,1).reshape(-1, C).detach()

        # squared euclidean distance
        z_sq  = (z_flat ** 2).sum(1, keepdim=True)
        cb_sq = (self.codebook ** 2).sum(1, keepdim=True).T
        dists = z_sq + cb_sq - 2 * z_flat @ self.codebook.T
        indices = dists.argmin(dim=1)

        z_q    = self.codebook[indices].reshape(B, D, H, W, C).permute(0,4,1,2,3)
        z_q_st = z + (z_q - z).detach()

        if self.training:
            with torch.no_grad():
                one_hot   = F.one_hot(indices, self.num_embeddings).float()
                self.cluster_size.mul_(self.decay).add_(one_hot.sum(0), alpha=1-self.decay)
                embed_sum = one_hot.T @ z_flat
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1-self.decay)

                n            = self.cluster_size.sum()
                cluster_size = (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
                new_cb       = self.embed_avg / cluster_size.unsqueeze(1)
                if not torch.isnan(new_cb).any() and new_cb.abs().max() < 10.0:
                    self.codebook.copy_(new_cb)

                dead = self.cluster_size < 1.0
                if dead.any():
                    n_dead     = int(dead.sum().item())
                    random_idx = torch.randint(0, z_flat.size(0), (n_dead,), device=z_flat.device)
                    noise      = torch.randn(n_dead, self.embedding_dim, device=z_flat.device) * 0.01
                    self.codebook[dead].copy_(z_flat[random_idx] + noise)
                    self.cluster_size[dead].fill_(1.0)
                    self.embed_avg[dead].copy_(self.codebook[dead])

        return z_q_st, z_q, indices.reshape(B, D, H, W)


class VQVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, latent_dim=64, num_embeddings=512):
        super().__init__()
        self.encoder   = Encoder(vocab_size, embed_dim, latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder   = Decoder(vocab_size, latent_dim)

    def forward(self, x):
        z                    = self.encoder(x)
        z_q_st, z_q, indices = self.quantizer(z)
        logits               = self.decoder(z_q_st)
        return logits, z, z_q, indices


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using: {device}")

    print("Loading data...")
    dataset = torch.load('data_files/chunks_filtered.pt').long()
    with open('data_files/vocab_filtered.json') as f:
        meta = json.load(f)
    vocab_size = len(meta['vocab'])
    print(f"Vocab size: {vocab_size}, Dataset: {dataset.shape}")

    n = int(0.9 * len(dataset))
    train_loader = DataLoader(TensorDataset(dataset[:n]), batch_size=8, shuffle=True)
    val_loader   = DataLoader(TensorDataset(dataset[n:]), batch_size=8, shuffle=False)

    # class weights
    print("Computing weights...")
    counts = torch.zeros(vocab_size)
    sample = dataset[torch.randint(0, len(dataset), (1000,))]
    for i in range(vocab_size):
        counts[i] = (sample == i).sum()

    air_idx   = next((i for i, n in enumerate(meta['vocab']) if n == 'air'),   0)
    stone_idx = next((i for i, n in enumerate(meta['vocab']) if n == 'stone'), None)

    weights       = torch.zeros(vocab_size)
    present       = counts > 0
    weights[present] = 1.0 / torch.sqrt(counts[present])
    weights[air_idx] = weights[present].min() * 0.1
    if stone_idx:
        weights[stone_idx] = weights[present].min() * 0.1
    weights = weights / weights.sum() * present.sum()
    weights = weights.to(device)

    model     = VQVAE(vocab_size=vocab_size, embed_dim=16, latent_dim=64, num_embeddings=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # scheduler steps per EPOCH not per batch
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    beta      = 0.1
    best_val  = float('inf')

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(40):
        model.train()
        total_loss = 0

        for (xb,) in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            xb = xb.to(device)
            optimizer.zero_grad()
            logits, z, z_q, indices = model(xb)
            recon_loss      = F.cross_entropy(logits, xb, weight=weights)
            commitment_loss = F.mse_loss(z, z_q.detach())
            loss = recon_loss + beta * commitment_loss
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # step scheduler once per epoch, not per batch
        scheduler.step()

        model.eval()
        val_loss       = 0
        correct        = 0
        total          = 0
        codebook_usage = set()

        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                logits, z, z_q, indices = model(xb)
                recon_loss      = F.cross_entropy(logits, xb, weight=weights)
                commitment_loss = F.mse_loss(z, z_q.detach())
                val_loss += (recon_loss + beta * commitment_loss).item()
                preds = logits.argmax(dim=1)
                correct += (preds == xb).sum().item()
                total   += xb.numel()
                codebook_usage.update(indices.cpu().numpy().flatten().tolist())

        avg_val = val_loss / len(val_loader)
        acc     = correct / total * 100
        print(f"Epoch {epoch+1}: train {total_loss/len(train_loader):.4f} val {avg_val:.4f} acc {acc:.2f}% codebook {len(codebook_usage)}/512")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), 'data_files/vqvae_best.pth')
            print(f"  Saved best!")
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved  = torch.cuda.memory_reserved()  / 1024**2
        print(f"  GPU memory: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")
        torch.cuda.empty_cache()


    print("Done")
