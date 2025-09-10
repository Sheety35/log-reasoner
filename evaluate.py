import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformer_model import LogTransformer, load_data, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS

def evaluate():
    # Load test data
    _, test_loader, vocab_size = load_data()

    # Load model
    model = LogTransformer(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS
    )
    model.load_state_dict(torch.load("transformer_model.pt"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = (outputs > 0.5).int().cpu().numpy()
            y_pred.extend(preds.flatten())
            y_true.extend(y_batch.numpy().flatten())

    print("📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, digits=4))

    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate()
