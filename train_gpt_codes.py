import torch
import torch.nn.functional as F
from tqdm import tqdm
from gpt import GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using: {device}")

codes = torch.load('data_files/codes.pt')
print(f"Codes shape: {codes.shape}")
print(f"Codebook entries used: {codes.unique().numel()}")

vocab_size = 513
block_size = 256

n           = int(0.9 * len(codes))
train_codes = codes[:n]
val_codes   = codes[n:]

def get_batch(split, batch_size=64):    
    data = train_codes if split == 'train' else val_codes
    ix   = torch.randint(len(data), (batch_size,))
    x    = data[ix].to(device)
    y    = torch.roll(x, -1, dims=1)
    y[:, -1] = 0
    return x, y

model = GPT(
    vocab_size = vocab_size,
    n_embd     = 256,
    n_heads    = 8,
    n_layers   = 6,
    block_size = block_size,
    dropout    = 0.1,
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
best_val  = float('inf')

for step in tqdm(range(5000)):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch('val')
            _, val_loss = model(xv, yv)
        model.train()
        print(f"step {step}: train {loss.item():.4f} val {val_loss.item():.4f}")
        if val_loss.item() < best_val:
            best_val = val_loss.item()
            torch.save(model.state_dict(), 'data_files/gpt_codes_best.pth')
            print(f"  Saved best!")

print("Done")
