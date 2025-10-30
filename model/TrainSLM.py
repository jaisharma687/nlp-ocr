import torch
from torch.utils.data import Dataset, DataLoader
import json
from model.Tokenizer import SimpleTokenizer
from model.Model import MiniTransformer
import torch.nn.functional as F
import os

class QADataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        with open(file_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(l) for l in f]
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = self._prepare_data()

    def _prepare_data(self):
        pairs = []
        for s in self.samples:
            text = f"Q: {s['prompt']} A: {s['response']}"
            encoded = self.tokenizer.encode(text)
            if len(encoded) > self.block_size:
                encoded = encoded[:self.block_size]
            pairs.append(torch.tensor(encoded, dtype=torch.long))
        return pairs

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# ✅ NEW: Collate function for dynamic padding
def pad_collate(batch):
    max_len = max([len(x) for x in batch])
    # PAD id will be filled later by caller to match tokenizer
    padded = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    return padded

def train_model(jsonl_path="data/qa_data.jsonl", epochs=10):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        all_text = f.read()
    tokenizer = SimpleTokenizer([all_text])

    dataset = QADataset(jsonl_path, tokenizer)
    def collate_with_pad(batch):
        padded = pad_collate(batch)
        # Replace default 0 with tokenizer.pad_id for clarity
        padded[padded == 0] = tokenizer.pad_id
        return padded

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_with_pad)

    model = MiniTransformer(vocab_size=tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_id
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "pad_id": tokenizer.pad_id,
        },
        "model/saved_model.pt"
    )
    print("✅ Model saved to model/saved_model.pt")

if __name__ == "__main__":
    train_model()
