import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ⚙️ Hyperparameters
SEQ_LEN = 10
EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_LAYERS = 2
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
VOCAB_SIZE = 200  # ← Will be updated based on dataset


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, src):
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True, average_attn_weights=False)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src, attn_weights


# 🧠 Transformer Model
class LogTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, embed_dim))

        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding
        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_weights_all.append(attn_weights)
        out = self.fc(x[:, -1, :])
        return self.sigmoid(out), attn_weights_all

def load_data(path="bgl_dataset.npz"):
    data = np.load(path)
    X_train = torch.tensor(data['X_train'], dtype=torch.long)
    y_train = torch.tensor(data['y_train'], dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(data['X_test'], dtype=torch.long)
    y_test = torch.tensor(data['y_test'], dtype=torch.float32).unsqueeze(-1)

    VOCAB_SIZE = int(X_train.max().item()) + 1

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)
    return train_loader, test_loader, VOCAB_SIZE

def train():
    train_loader, test_loader, vocab_size = load_data()

    model = LogTransformer(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN 
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output, _ = model(X_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "transformer_model.pt")
    print("✅ Model saved as transformer_model.pt")

if __name__ == "__main__":
    train()
