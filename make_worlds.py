import torch
import json
import numpy as np
import os
import random
from vqvae import VQVAE
from gpt import GPT
import matplotlib.pyplot as plt
from tqdm import tqdm

NUM_SAMPLES = 300
TEMPERATURE = 1.0
OUTPUT_DIR = "generated_worlds"
SEED = None               

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if SEED is not None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

with open('data_files/vocab_filtered.json') as f:
    meta = json.load(f)

id2block = {i: b for i, b in enumerate(meta['vocab'])}

vqvae = VQVAE(vocab_size=31, embed_dim=16, latent_dim=64, num_embeddings=512).to(device)
vqvae.load_state_dict(torch.load('data_files/vqvae_best.pth', map_location=device))
vqvae.eval()

gpt = GPT(vocab_size=513, n_embd=256, n_heads=8, n_layers=6, block_size=256, dropout=0.1).to(device)
gpt.load_state_dict(torch.load('data_files/gpt_codes_best.pth', map_location=device))
gpt.eval()

@torch.no_grad()
def generate_chunk():
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    codes_seq = gpt.generate(context, max_new_tokens=256, temperature=TEMPERATURE)

    codes = codes_seq[:, 1:257].reshape(1, 4, 4, 16)

    z_q = vqvae.quantizer.codebook[codes]
    z_q = z_q.permute(0, 4, 1, 2, 3).float()

    vox_logits = vqvae.decoder(z_q)
    chunk = vox_logits.argmax(dim=1)[0].cpu().numpy()
    return chunk

def render_voxels(grid):
    COLORS = {
        'air': (0, 0, 0, 0),
        'cave_air': (0, 0, 0, 0),
        'stone': (0.5, 0.5, 0.5, 1),
        'deepslate': (0.2, 0.2, 0.2, 1),
        'dirt': (0.55, 0.27, 0.07, 1),
        'grass_block': (0.35, 0.6, 0.2, 1),
        'water': (0.2, 0.4, 0.8, 0.5),
        'sand': (0.85, 0.8, 0.5, 1),
        'sandstone': (0.8, 0.75, 0.45, 1),
        'gravel': (0.55, 0.55, 0.55, 1),
        'granite': (0.6, 0.45, 0.4, 1),
        'diorite': (0.75, 0.75, 0.75, 1),
        'andesite': (0.52, 0.52, 0.52, 1),
        'tuff': (0.4, 0.4, 0.35, 1),
        'clay': (0.6, 0.65, 0.75, 1),
        'coal_ore': (0.15, 0.15, 0.15, 1),
        'iron_ore': (0.75, 0.65, 0.55, 1),
        'copper_ore': (0.7, 0.45, 0.35, 1),
        'lapis_ore': (0.1, 0.3, 0.7, 1),
        'dripstone_block': (0.5, 0.4, 0.3, 1),
        'pointed_dripstone': (0.45, 0.35, 0.25, 1),
        'smooth_basalt': (0.3, 0.3, 0.3, 1),
        'ice': (0.6, 0.8, 1.0, 0.6),
        'packed_ice': (0.5, 0.7, 1.0, 1),
        'snow': (0.95, 0.95, 0.95, 1),
        'snow_block': (1.0, 1.0, 1.0, 1),
        'oak_leaves': (0.1, 0.4, 0.1, 0.7),
        'birch_leaves': (0.3, 0.5, 0.2, 0.7),
        'spruce_leaves': (0.05, 0.3, 0.1, 0.7),
        'leaf_litter': (0.4, 0.3, 0.1, 1),
        'moss_block': (0.3, 0.4, 0.1, 1),
        'other': (1, 0, 1, 1) 
    }

    grid_fixed = grid.transpose(0, 1, 2)
    
    X_max, Z_max, Y_max = grid_fixed.shape
    voxels = np.zeros((X_max, Z_max, Y_max), dtype=bool)
    facecolors = np.zeros((X_max, Z_max, Y_max, 4))

    for x in range(X_max):
        for z in range(Z_max):
            for y in range(Y_max - 8):
                block_id = int(grid_fixed[x, z, y])
                name = id2block.get(block_id, 'other')
                
                if name == 'air': 
                    continue

                voxels[x, z, y] = True
                facecolors[x, z, y] = COLORS.get(name, COLORS['other'])

    return voxels, facecolors

print(f"Generating {NUM_SAMPLES} worlds...")

for i in tqdm(range(NUM_SAMPLES)):
    chunk = generate_chunk()
    voxels, colors = render_voxels(chunk)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(voxels, facecolors=colors, edgecolor=None)
    ax.view_init(elev=30, azim=45)
    ax.set_axis_off()

    filename = os.path.join(OUTPUT_DIR, f"world_{i:04d}.png")
    plt.savefig(filename, dpi=150)
    plt.close(fig) 

print(f"Done. Saved to {OUTPUT_DIR}/")