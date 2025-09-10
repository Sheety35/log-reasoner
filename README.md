# 🧠 Log Reasoner — Log Anomaly Detection using Transformer + LLM Explanations

This project implements an end-to-end **log anomaly detection pipeline** using:
- **Transformer**-based neural models for anomaly classification
- **Attention visualization** and **LLM-generated explanations** for model interpretability

---

## 📂 Project Structure

```
.
├── transformer_model.py          # Train Transformer-based classifier
├── attention_model.py            # Explain model predictions using attention & LLM
├── bgl_dataset.npz              # Dataset (input sequences + labels)
├── transformer_model.pt         # Saved Transformer model
├── eventid_encoder.pkl          # Encoder for converting log IDs
├── event_template_lookup.pkl    # Mapping token IDs → templates
```

---

## 📈 Dataset Format

The dataset file `bgl_dataset.npz` should contain the following NumPy arrays:
- `X_train`, `y_train`: Training sequences and labels
- `X_test`, `y_test`: Test sequences and labels

Each sequence is of length `10` (configurable), where each element is an integer token ID for the Transformer model.

---

## 🚀 Quick Start

### ▶️ Train Transformer Model
```bash
python transformer_model.py
```
This trains a multi-head Transformer model and saves it as `transformer_model.pt`.

### 🤖 Explain Predictions using LLM
1. Install Ollama and ensure a model like `mistral` is available locally
2. Then run:
```bash
python attention_model.py
```

The script will:
- Load the Transformer model
- Pick an anomalous log sequence
- Visualize attention weights (optional)
- Use an LLM to explain the model's decision

---

## 🛠 Features

✅ Custom Transformer Encoder with attention extraction  
✅ Log token decoding using `eventid_encoder.pkl` and `event_template_lookup.pkl`  
✅ LLM integration (via Ollama CLI) for natural language explanations  
✅ Exportable predictions & explanations to Excel (`model_explanations.xlsx`)  

---

## 🧪 Available Functions in attention_model.py

- `explain(index=0)`: Explain a specific sample by index
- `evaluate_multiple_samples(num_samples=20)`: Loop through multiple test samples and explain them
- `export_predictions_to_excel(...)`: Save predictions & explanations in Excel format
- `explain_anomalous_sample()`: Find and explain the first anomaly in the test set
- `print_label_distribution()`: Print label counts (normal vs. anomaly)

---

## 🧠 Example LLM Prompt

The LLM receives a prompt like:

```
You are a log anomaly detection assistant.

Given the input log messages: ['EVENT_1', 'EVENT_2', ..., 'EVENT_10']
The model predicted anomaly label: 1 (1 = anomaly, 0 = normal).

Explain why this log sequence is considered anomalous.
(Optional) Attention weights: [[0.3, 0.2, ...]]

Be concise, clear, and explain any unusual patterns.
```

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- NumPy
- pandas
- matplotlib
- Ollama (for local LLM inference)

Install dependencies:
```bash
pip install torch numpy pandas matplotlib
```

---

## 📎 Notes

- The Transformer model expects integer token sequences
- Make sure your `.npz` dataset format matches the expected shapes and types
- Ensure Ollama is running locally with your preferred model loaded

---

## 🧑‍💻 Author

**Sheety35** — 2025  
GitHub: [Sheety35](https://github.com/Sheety35)

---

## 📄 License

MIT License
