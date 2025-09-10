import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

SEQ_LEN = 10  # Window size

def load_and_preprocess(path):
    df = pd.read_csv(path)
    
    # Filter labeled data
    df = df[df['Label'].notnull()]
    
    # Binary label: 0 = normal, 1 = anomaly
    df['is_anomaly'] = (df['Label'] != '-').astype(int)

    # Encode EventId (or EventTemplate for better generalization)
    encoder = LabelEncoder()
    df['EventId_enc'] = encoder.fit_transform(df['EventTemplate'])
    # Save encoder
    with open("eventid_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    # Also save mapping to human-readable messages
    with open("event_template_lookup.pkl", "wb") as f:
        pickle.dump({i: s for i, s in enumerate(encoder.classes_)}, f)

    return df, encoder

def create_sequences(df, seq_len):
    sequences = []
    labels = []

    for i in range(len(df) - seq_len):
        seq = df['EventId_enc'].iloc[i:i+seq_len].tolist()
        label = df['is_anomaly'].iloc[i+seq_len]
        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)

if __name__ == "__main__":
    df, enc = load_and_preprocess("BGL_2k.log_structured.csv")
    X, y = create_sequences(df, SEQ_LEN)

    # Split for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples: {len(X_test)}")

    # Save for model
    np.savez("bgl_dataset.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
