import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Detektor Spam Judi Online",
    page_icon="📧",
    layout="centered"
)

# --- Fungsi untuk Memuat Model dari Hugging Face Hub ---
@st.cache_resource
def load_model():
    """Memuat model dan tokenizer dari Hugging Face Hub."""
    # Langsung menggunakan model yang sudah Anda unggah
    model_name = "jelialmutaali/online-gambling-spam-detector"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Gagal memuat model dari Hugging Face Hub: {e}")
        st.info("Pastikan nama repositori sudah benar dan modelnya bersifat public.")
        return None, None, None

# --- Fungsi Preprocessing Teks ---
def clean_text(text):
    """Membersihkan teks input."""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# --- Muat Model ---
tokenizer, model, device = load_model()

# --- Antarmuka (UI) Aplikasi Streamlit ---
st.title("📧 Detektor Spam Judi Online")
st.write(
    "Aplikasi ini menggunakan model `jelialmutaali/online-gambling-spam-detector` untuk "
    "mendeteksi apakah sebuah komentar mengandung spam atau tidak."
)
st.markdown("---")

# Hanya lanjutkan jika model berhasil dimuat
if model and tokenizer:
    user_input = st.text_area(
        "Masukkan komentar untuk dianalisis:", 
        "wd terus nih di situs gacor, cek bio!", 
        height=150
    )

    if st.button("Deteksi Komentar"):
        if user_input and user_input.strip() != "":
            # 1. Bersihkan teks input
            cleaned_text = clean_text(user_input)

            # 2. Tokenisasi teks (menggunakan 'pt' untuk PyTorch)
            inputs = tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            inputs = {key: val.to(device) for key, val in inputs.items()}

            # 3. Lakukan Prediksi
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                prediction_idx = torch.argmax(logits, dim=1).item()

            # 4. Tampilkan Hasil
            st.subheader("Hasil Analisis:")
            if prediction_idx == 1:
                st.error("🚨 Terdeteksi sebagai **Spam**.")
                st.balloons()
            else:
                st.success("✅ Terdeteksi sebagai **Bukan Spam**.")
        else:
            st.warning("Mohon masukkan teks komentar terlebih dahulu.")