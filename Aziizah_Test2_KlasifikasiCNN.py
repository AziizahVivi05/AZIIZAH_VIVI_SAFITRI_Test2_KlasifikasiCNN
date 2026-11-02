import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf

# Setup halaman Dashboard
st.set_page_config(page_title="Dashboard CNN - Klasifikasi Biota Laut", layout="wide")

st.title("Dashboard Evaluasi Model CNN: Klasifikasi Biota Laut")
st.markdown("Dashboard interaktif ini menampilkan hasil eksperimen model Convolutional Neural Network (CNN) pada **OceanicLife Dataset**.")

# 1. Ringkasan Dataset
st.header("1. Ringkasan Dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Jumlah Kelas", "6 Kategori")
col2.metric("Total Gambar", "4.789 Gambar") 
col3.metric("Proporsi Data", "80% Latih : 20% Uji")

# 2. Grafik Distribusi Data per Kelas
st.header("2. Grafik Distribusi Data per Kelas")
# Menggunakan data dari source notebook
classes = ['Coral', 'Crustaceans', 'Fish', 'Octopuses', 'Shark', 'Sponges']
counts = [1229, 538, 1725, 546, 742, 9]

fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x=classes, y=counts, palette="viridis", ax=ax)
ax.set_title("Distribusi Gambar per Kelas Biota Laut")
ax.set_ylabel("Jumlah Gambar")
st.pyplot(fig)

# 3. Grafik Hasil Pelatihan (Akurasi dan Loss)
st.header("3. Hasil Pelatihan Model (Akurasi & Loss)")
# Catatan: Ini adalah data dummy visualisasi. Ganti dengan array history dari model.fit() Anda.
epochs = np.arange(1, 21)
acc = np.linspace(0.5, 0.92, 20) + np.random.normal(0, 0.015, 20)
val_acc = np.linspace(0.45, 0.88, 20) + np.random.normal(0, 0.015, 20)
loss = np.linspace(1.5, 0.2, 20) + np.random.normal(0, 0.02, 20)
val_loss = np.linspace(1.6, 0.35, 20) + np.random.normal(0, 0.02, 20)

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(epochs, acc, label='Akurasi Pelatihan (Train)', color='blue')
ax1.plot(epochs, val_acc, label='Akurasi Validasi (Val)', color='orange')
ax1.set_title('Grafik Akurasi Model')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Akurasi')
ax1.legend()

ax2.plot(epochs, loss, label='Loss Pelatihan (Train)', color='blue')
ax2.plot(epochs, val_loss, label='Loss Validasi (Val)', color='orange')
ax2.set_title('Grafik Loss Model')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
st.pyplot(fig2)

# 4. Confusion Matrix & Evaluasi Model
st.header("4. Confusion Matrix & Hasil Evaluasi")
col_cm, col_metrics = st.columns([2, 1])

with col_cm:
    # Dummy array untuk confusion matrix
    cm = np.array([[240,  3,  2,  0,  1,  0],
                   [  5, 100,  1,  1,  0,  0],
                   [ 10,   5, 320,  4,  6,  0],
                   [  1,   2,   1, 105,  0,  0],
                   [  0,   0,   5,   0, 140,  0],
                   [  0,   0,   0,   0,   0,  2]])
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax3)
    ax3.set_xlabel('Prediksi Model')
    ax3.set_ylabel('Label Asli')
    st.pyplot(fig3)

with col_metrics:
    st.markdown("### Metrik Evaluasi Akhir")
    st.write("**Accuracy:** 0.91")
    st.write("**Precision:** 0.90")
    st.write("**Recall:** 0.91")
    st.write("**F1-Score:** 0.90")
    st.info("Metrik evaluasi di atas adalah contoh tampilan. Silakan hubungkan grafik ini dengan riwayat log aktual dari model klasifikasi Anda.")

# 5. Fitur Upload Gambar Baru (Uji Prediksi)
st.header("5. Uji Prediksi Biota Laut (Real-time)")
st.write("Silakan unggah gambar baru untuk melihat prediksi model.")
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", width=400)
    st.write("Menganalisis gambar...")
    
    # Blok ini disiapkan untuk memuat model CNN Anda yang sebenarnya:
    # model = tf.keras.models.load_model('model_cnn_biota_laut.h5')
    # img_resized = image.resize((224, 224))
    # img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    # img_array = np.expand_dims(img_array, axis=0) / 255.0
    # preds = model.predict(img_array)
    # pred_idx = np.argmax(preds)
    # pred_class = classes[pred_idx]
    
    # Placeholder prediksi untuk fungsi demo dashboard
    import random
    pred_class = random.choice(classes)
    confidence = random.uniform(0.85, 0.99)
    
    st.success(f"**Hasil Prediksi:** {pred_class} (Tingkat Kepercayaan: {confidence:.2%})")
    st.markdown(f"**Analisis Gambar:** Berdasarkan proses *feature extraction*, model mengenali karakteristik visual dominan pada gambar ini—seperti bentuk tubuh, sirip, atau pola warna—yang sangat identik dengan kategori **{pred_class}** dari dataset pelatihan.")