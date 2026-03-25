import torch
import json
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from vqvae import VQVAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = torch.load('data_files/chunks_filtered.pt').long()
with open('data_files/vocab_filtered.json') as f:
    meta = json.load(f)
vocab_size = len(meta['vocab'])

model = VQVAE(vocab_size=vocab_size, embed_dim=16, latent_dim=64, num_embeddings=512).to(device)
model.load_state_dict(torch.load('data_files/vqvae_best.pth'))
model.eval()

all_codes = []
loader    = DataLoader(TensorDataset(dataset), batch_size=8)

with torch.no_grad():
    for (xb,) in tqdm(loader, desc="Extracting codes"):
        xb = xb.to(device)
        _, _, _, indices = model(xb)
        all_codes.append(indices.cpu())

all_codes = torch.cat(all_codes)
all_codes = all_codes.reshape(len(all_codes), -1) 
torch.save(all_codes, 'data_files/codes.pt')
print(f"Saved codes: {all_codes.shape}")
print(f"Codebook usage: {all_codes.unique().numel()}/512 entries used")
