import subprocess
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_model import LogTransformer, load_data, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS
import pickle

with open("eventid_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("event_template_lookup.pkl", "rb") as f:
    id_to_template = pickle.load(f)

# Reverse mapping from token ID back to EventId
id_to_log = {i: label for i, label in enumerate(encoder.classes_)}

def query_ollama(prompt: str, model_name="mistral:latest"):
    """
    Send a prompt to ollama CLI and get generated text using stdin to avoid command-line issues.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error querying ollama: {e}")
        print(f"stderr: {e.stderr}")
        return None


def explain(index=0):
    _, test_loader, vocab_size = load_data()
    test_data = list(test_loader.dataset)
    SEQ_LEN = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LogTransformer(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN
    )
    model.load_state_dict(torch.load("transformer_model.pt"))
    model.to(device)
    model.eval()

    # Pick a sample and prepare input
    x, y = test_data[index]
    x = x.unsqueeze(0).to(device)  # add batch dimension

    # Run model and get attention weights and prediction
    with torch.no_grad():
        output, attn_weights_all = model(x)  # attn_weights_all is a list: one tensor per layer

    predicted_label = int(output.item() > 0.5)

    print(f"True Label: {int(y)}")
    print(f"Predicted Label: {predicted_label}")

    # Visualize attention (optional, your existing code)
    attention = attn_weights_all[0].squeeze(0).cpu().numpy()
    """for head in range(attention.shape[0]):
        plt.figure(figsize=(8, 6))
        plt.imshow(attention[head], cmap='viridis', interpolation='nearest')
        plt.title(f'Attention Head {head+1} - Layer 1')
        plt.xlabel('Key Token Index')
        plt.ylabel('Query Token Index')
        plt.colorbar()
        plt.tight_layout()
        plt.show()"""

    # Prepare prompt for LLM explanation
    input_token_ids = x.squeeze(0).cpu().tolist()
    decoded_tokens = [id_to_log.get(tok, f"<UNK:{tok}>") for tok in input_token_ids]
    print(f"Token IDs: {input_token_ids}")
    print(f"Decoded Log Events: {decoded_tokens}")
    prompt = (
        f"You are a log anomaly detection assistant.\n"
        f"Given the input log messages: {decoded_tokens}\n"
        f"The model predicted anomaly label: {predicted_label} (1 = anomaly, 0 = normal).\n"
        f"Explain why this log sequence is considered {'anomalous' if predicted_label else 'normal'}.\n"
        f"(Optional) Attention weights: {attention[0].round(3).tolist()}\n"
        f"Be concise, clear, and explain any unusual patterns you detect."
    )

    # Query the LLM
    explanation = query_ollama(prompt)
    print("\n--- LLM Explanation ---")
    print(explanation)

def evaluate_multiple_samples(num_samples=20):
    _, test_loader, vocab_size = load_data()
    test_data = list(test_loader.dataset)
    SEQ_LEN = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LogTransformer(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN
    )
    model.load_state_dict(torch.load("transformer_model.pt"))
    model.to(device)
    model.eval()

    for index in range(min(num_samples, len(test_data))):
        x, y = test_data[index]
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            output, attn_weights_all = model(x)

        predicted_label = int(output.item() > 0.5)
        true_label = int(y.item())

        print(f"Sample {index}: True Label: {true_label}, Predicted Label: {predicted_label}")

        # Prepare prompt for LLM explanation
        input_tokens = x.squeeze(0).cpu().tolist()
        attention = attn_weights_all[0].squeeze(0).cpu().numpy()
        prompt = (
            f"You are a log anomaly detection assistant.\n"
            f"Given the input log token IDs: {input_tokens}\n"
            f"The model predicted anomaly label: {predicted_label} (1 means anomaly, 0 means normal).\n"
            f"Explain in simple language why this log is considered {'anomalous' if predicted_label == 1 else 'normal'}.\n"
            f"Use the attention weights from the first attention head of the first layer to help your explanation if needed.\n"
            f"Attention weights:\n{attention[0].round(3).tolist()}\n"
            f"Please provide a concise explanation:"
        )

        explanation = query_ollama(prompt)
        print("\n--- LLM Explanation ---")
        print(explanation)
        print("-" * 60)

def export_predictions_to_excel(num_samples=20, output_file="model_explanations.xlsx"):
    _, test_loader, vocab_size = load_data()
    test_data = list(test_loader.dataset)
    SEQ_LEN = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LogTransformer(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN
    )
    model.load_state_dict(torch.load("transformer_model.pt"))
    model.to(device)
    model.eval()

    records = []

    for index in range(min(num_samples, len(test_data))):
        x, y = test_data[index]
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            output, attn_weights_all = model(x)

        pred_prob = output.item()
        predicted_label = int(pred_prob > 0.5)
        true_label = int(y.item())

        input_tokens = x.squeeze(0).cpu().tolist()
        attention = attn_weights_all[0].squeeze(0).cpu().numpy()

        prompt = (
            f"You are a log anomaly detection assistant.\n"
            f"Given the input log token IDs: {input_tokens}\n"
            f"The model predicted anomaly label: {predicted_label} (1 means anomaly, 0 means normal).\n"
            f"Explain in simple language why this log is considered {'anomalous' if predicted_label == 1 else 'normal'}.\n"
            f"Use the attention weights from the first attention head of the first layer to help your explanation if needed.\n"
            f"Attention weights:\n{attention[0].round(3).tolist()}\n"
            f"Please provide a concise explanation:"
        )

        explanation = query_ollama(prompt)

        records.append({
            "Index": index,
            "True Label": true_label,
            "Predicted Label": predicted_label,
            "Predicted Probability": round(pred_prob, 4),
            "Token IDs": str(input_tokens),
            "LLM Explanation": explanation
        })

    df = pd.DataFrame(records)
    df.to_excel(output_file, index=False)
    print(f"✅ Excel report saved to {output_file}")

def explain_anomalous_sample():
    _, test_loader, vocab_size = load_data()
    test_data = list(test_loader.dataset)
    SEQ_LEN = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LogTransformer(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN
    )
    model.load_state_dict(torch.load("transformer_model.pt", weights_only=True))
    model.to(device)
    model.eval()

    for index, (x, y) in enumerate(test_data):
        if y.item() == 1:
            print(f"Found anomaly at index {index}")
            x = x.unsqueeze(0).to(device)

            with torch.no_grad():
                output, attn_weights_all = model(x)

            predicted_label = int(output.item() > 0.5)
            print(f"True Label: 1")
            print(f"Predicted Label: {predicted_label}")

            attention = attn_weights_all[0].squeeze(0).cpu().numpy()
            input_token_ids = x.squeeze(0).cpu().tolist()
            decoded_tokens = [id_to_log.get(tok, f"<UNK:{tok}>") for tok in input_token_ids]
            decoded_templates = [id_to_template.get(tok, f"<UNK:{tok}>") for tok in input_token_ids]

            # ✅ Add print here to confirm decoding works
            print(f"Token IDs: {input_token_ids}")
            print(f"Decoded Log Events: {decoded_tokens}")
            print(f"Decoded Log Events: {decoded_templates}")

            prompt = (
                f"You are a log anomaly detection assistant.\n"
                f"Given the input log messages:\n"
                f"{decoded_templates}\n"
                f"The model predicted anomaly label: {predicted_label} (1 = anomaly, 0 = normal).\n"
                f"Explain why this log sequence is considered {'anomalous' if predicted_label else 'normal'}.\n"
                f"(Optional) Attention weights: {attention[0].round(3).tolist()}\n"
                f"Be concise, clear, and explain any unusual patterns you detect."
            )

            explanation = query_ollama(prompt)
            print("\n--- LLM Explanation ---")
            print(explanation)
            break
    else:
        print("❌ No anomalous samples (label == 1) found in the test set.")


def print_label_distribution():
    _, test_loader, _ = load_data()
    test_data = list(test_loader.dataset)

    labels = [int(y.item()) for _, y in test_data]
    num_anomalies = sum(labels)
    print(f"Total samples: {len(labels)}")
    print(f"Normal (0): {len(labels) - num_anomalies}")
    print(f"Anomalous (1): {num_anomalies}")
    
if __name__ == "__main__":
    print_label_distribution()
    explain_anomalous_sample()