import torch
import json
import numpy as np
from vqvae import VQVAE
from gpt import GPT
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
def generate_chunk(temperature=0.8):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    codes_seq = gpt.generate(context, max_new_tokens=256, temperature=temperature)
    codes = codes_seq[:, 1:257].reshape(1, 16, 4, 4)
    
    z_q = vqvae.quantizer.codebook[codes] 
    z_q = z_q.permute(0, 4, 1, 2, 3).float()
    
    vox_logits = vqvae.decoder(z_q)
    chunk = vox_logits.argmax(dim=1)[0].cpu().numpy()
    return chunk

def render_voxels(grid):
    COLORS = {
        'air': (0, 0, 0, 0),
        'stone': (0.5, 0.5, 0.5, 1),
        'deepslate': (0.2, 0.2, 0.2, 1),
        'dirt': (0.55, 0.27, 0.07, 1),
        'grass_block': (0.13, 0.55, 0.13, 1),
        'water': (0.12, 0.56, 1.0, 0.5),
        'sand': (0.9, 0.8, 0.6, 1),
        'oak_log': (0.3, 0.2, 0.1, 1),
        'oak_leaves': (0, 0.4, 0, 0.8),
        'other': (1, 0, 1, 1)
    }

    Y_dim, X_dim, Z_dim = grid.shape
    voxels = np.zeros((X_dim, Z_dim, Y_dim), dtype=bool)
    facecolors = np.zeros((X_dim, Z_dim, Y_dim, 4))

    for y in range(Y_dim):
        for x in range(X_dim):
            for z in range(Z_dim):
                block_id = int(grid[y, x, z])
                name = id2block.get(block_id, 'other')
                if name == 'air': continue

                if y < Y_dim - 1:
                    above_id = int(grid[y+1, x, z])
                    if id2block.get(above_id, 'air') != 'air':
                        continue

                voxels[x, z, y] = True
                facecolors[x, z, y] = COLORS.get(name, COLORS['other'])

    return voxels, facecolors

def render_slice(grid, z_slice=16):
    COLORS = {
        'air': '#87CEEB', 'stone': '#808080', 'deepslate': '#3a3a3a',
        'dirt': '#8B4513', 'grass_block': '#228B22', 'water': '#1E90FF',
        'granite': '#C47A5A', 'andesite': '#A0A0A0', 'diorite': '#D8D8D8',
        'gravel': '#9A9A8A', 'coal_ore': '#1a1a1a', 'oak_leaves': '#006400',
        'other': '#FF00FF'
    }
    
    img = np.zeros((128, 32, 3))
    
    for y in range(128):
        for x in range(32):
            block_id = int(grid[y, x, z_slice])
            name = id2block.get(block_id, 'other')
            hex_c = COLORS.get(name, COLORS['other'])
            
            rgb = tuple(int(hex_c[i:i+2], 16)/255 for i in (1,3,5))
            img[127-y, x] = rgb
    return img

print("Generating chunks...")
fig = plt.figure(figsize=(25, 6))

for i in range(5):
    ax = fig.add_subplot(1, 5, i+1, projection='3d')

    chunk = generate_chunk(temperature=1.0)
    voxels, colors = render_voxels(chunk)

    # img = render_slice(chunk, z_slice=16)
    # ax.imshow(img)
    # ax.set_title(f"Generated {i+1}")
    # ax.axis('off')

    ax.voxels(voxels, facecolors=colors, edgecolor=None)
    ax.view_init(elev=30, azim=45)
    ax.set_title(f'Generated {i+1}')
    ax.set_axis_off()

plt.suptitle('Generated Minecraft Chunks (VQ-VAE + GPT)', fontsize=14)
plt.tight_layout()
plt.savefig('generated_chunks_voxels.png', dpi=150)
print("Saved generated_chunks_voxels.png")