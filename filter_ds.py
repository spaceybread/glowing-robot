import torch
import json
from collections import Counter
from tqdm import tqdm

dataset = torch.load('data_files/chunks.pt')
with open('data_files/vocab.json') as f:
    meta = json.load(f)

print("Counting blocks...")
counts = Counter()
for i in tqdm(range(len(dataset))):
    ids, cnts = dataset[i].unique(return_counts=True)
    for id_, cnt in zip(ids.tolist(), cnts.tolist()):
        counts[meta['vocab'][id_]] += cnt

top30 = [name for name, _ in counts.most_common(30)]
if 'air' not in top30:
    top30.append('air')
print("Keeping:", top30)

new_vocab = sorted(top30)
old2new = {}
other_idx = len(new_vocab)
new_vocab.append('other')

for old_idx, name in enumerate(meta['vocab']):
    if name in new_vocab:
        old2new[old_idx] = new_vocab.index(name)
    else:
        old2new[old_idx] = other_idx

print("Remapping dataset...")
mapping = torch.zeros(len(meta['vocab']), dtype=torch.long)
for old, new in old2new.items():
    mapping[old] = new

new_dataset = mapping[dataset.reshape(-1).long()].reshape(dataset.shape)
print(f"New vocab size: {len(new_vocab)}")
print(f"Dataset shape: {new_dataset.shape}")

torch.save(new_dataset, 'data_files/chunks_filtered.pt')
with open('data_files/vocab_filtered.json', 'w') as f:
    json.dump({'vocab': new_vocab}, f)

print("Saved chunks_filtered.pt and vocab_filtered.json")