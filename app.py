import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import requests
import pickle
import os

st.set_page_config(page_title="Deteksi Komentar SARA", layout="wide")
st.title("🧠 Deteksi Komentar SARA")
st.markdown("Model yang digunakan: **IndoBERT Base** dan **BiLSTM**")

# === MODEL 1: IndoBERT BASE ===
@st.cache_resource
def load_indobert_base():
    model_name = "Ricky131/indobert-base-sara-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        use_safetensors=False  # ⛔️ wajib untuk load model .bin
    )
    model.eval()
    return tokenizer, model

# === MODEL 2: BiLSTM ===
@st.cache_resource
def load_bilstm_model():
    model_url = "https://huggingface.co/Ricky131/bilstm-sara-detector/resolve/main/bilstm_model.pth"
    vocab_url = "https://huggingface.co/Ricky131/bilstm-sara-detector/resolve/main/vocab.pkl"

    os.makedirs("bilstm_temp", exist_ok=True)
    model_path = "bilstm_temp/bilstm_model.pth"
    vocab_path = "bilstm_temp/vocab.pkl"

    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            f.write(requests.get(model_url).content)
    if not os.path.exists(vocab_path):
        with open(vocab_path, "wb") as f:
            f.write(requests.get(vocab_url).content)

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, output_dim=2):
            super(BiLSTMClassifier, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
            self.dropout = nn.Dropout(0.3)

        def forward(self, input_ids):
            embedded = self.embedding(input_ids)
            _, (hidden, _) = self.lstm(embedded)
            out = torch.cat((hidden[0], hidden[1]), dim=1)
            out = self.dropout(out)
            return self.fc(out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return vocab, model

# === FUNGSI PREDIKSI ===
def predict_indobert_base(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()
    return pred, confidence

def predict_bilstm(text, vocab, model):
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import word_tokenize

    max_len = 100
    tokens = word_tokenize(text)
    encoded = [vocab.get(word, vocab.get('<UNK>', 1)) for word in tokens]
    if len(encoded) < max_len:
        encoded += [vocab.get('<PAD>', 0)] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]

    input_tensor = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()
    return pred, confidence

# === UI ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("🟡 IndoBERT Base")
    tokenizer_base, model_base = load_indobert_base()
    input_text1 = st.text_area("Masukkan komentar untuk IndoBERT Base", key="input1")
    if st.button("Deteksi dengan IndoBERT Base"):
        label, conf = predict_indobert_base(input_text1, tokenizer_base, model_base)
        st.success(f"Prediksi: {'SARA' if label==1 else 'TIDAK SARA'} (Confidence: {conf:.2f})")

with col2:
    st.subheader("🔵 BiLSTM")
    vocab_bilstm, model_bilstm = load_bilstm_model()
    input_text2 = st.text_area("Masukkan komentar untuk BiLSTM", key="input2")
    if st.button("Deteksi dengan BiLSTM"):
        label, conf = predict_bilstm(input_text2, vocab_bilstm, model_bilstm)
        st.success(f"Prediksi: {'SARA' if label==1 else 'TIDAK SARA'} (Confidence: {conf:.2f})")
