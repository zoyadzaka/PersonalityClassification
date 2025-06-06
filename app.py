import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

# Load model
model = joblib.load("model.pkl")

# ====== SIDEBAR ======
st.sidebar.title("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV", type=["csv"])

# Tambahkan informasi di sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Tentang Aplikasi")
st.sidebar.markdown("""
Aplikasi ini memprediksi kepribadian seseorang sebagai **Extrovert** atau **Introvert** 
berdasarkan data perilaku sosial mereka.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Fitur yang Dianalisis")
st.sidebar.markdown("""
- **Time_spent_Alone**: Waktu yang dihabiskan sendiri (jam/hari)
- **Stage_fear**: Ketakutan berbicara di depan umum (Ya/Tidak)
- **Social_event_attendance**: Frekuensi menghadiri acara sosial (1-10)
- **Going_outside**: Frekuensi keluar rumah (1-10)
- **Drained_after_socializing**: Merasa lelah setelah bersosialisasi (Ya/Tidak)
- **Friends_circle_size**: Jumlah teman dekat
- **Post_frequency**: Frekuensi posting di media sosial (1-10)
""")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Panduan Penggunaan")
st.sidebar.markdown("""
1. Unggah file CSV yang berisi data perilaku sosial
2. Pastikan file memiliki semua kolom yang diperlukan
3. Sistem akan otomatis melakukan prediksi
4. Lihat hasil dan visualisasi di panel utama
5. Unduh hasil prediksi jika diperlukan
""")

st.sidebar.markdown("---")
st.sidebar.subheader("👥 Tentang Kami")
st.sidebar.markdown("""
Dikembangkan oleh **Kelompok 86**
- Chantika Anasthya #1301223321
- Nabila Keiko Aura Pasha #1301220267
- Muhammad Zoyadzaka Wicaksono #1301220471

Sebagai bagian dari Tugas Besar Machine Learning
""")

# Tambahkan ekspander untuk informasi tambahan
with st.sidebar.expander("📊 Informasi Model"):
    st.markdown("""
    Membangun Model Klasifikasi Kepribadian (Introvert atau Ekstrovert) Menggunakan Metode Supervised Learning
     dengan Algoritma Support Vector Machine, Random Forest, dan Gradient Boosting.
      
    """)

# ====== HEADER UTAMA ======
st.title("🧑‍🤝‍🧑 Aplikasi Prediksi Kepribadian")
st.markdown("Prediksi apakah seseorang termasuk **Extrovert** atau **Introvert** berdasarkan data perilaku sosial.")
st.markdown("---")

# ====== LOAD DATA ======
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File berhasil diunggah dan dibaca!")
    
    st.subheader("📋 Data yang Diupload")
    st.dataframe(df.head())

    # Kolom yang digunakan
    feature_columns = [
        'Time_spent_Alone',
        'Stage_fear',
        'Social_event_attendance',
        'Going_outside',
        'Drained_after_socializing',
        'Friends_circle_size',
        'Post_frequency'
    ]
    cat_cols = ['Stage_fear', 'Drained_after_socializing']

    # Cek fitur
    if all(col in df.columns for col in feature_columns):
        # Label encoding kategori Yes/No
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        X = df[feature_columns]
        
        if X.isnull().values.any():
            st.warning("⚠️ Data mengandung nilai kosong (NaN), akan diisi otomatis dengan nilai median.")
            X = X.fillna(X.median(numeric_only=True))  
        predictions = model.predict(X)
        df['Prediksi'] = predictions

        st.subheader("📊 Hasil Prediksi")
        st.dataframe(df)

        # Grafik distribusi hasil prediksi
        st.subheader("📈 Distribusi Hasil Prediksi")
        pred_counts = df['Prediksi'].value_counts()
        fig_pred, ax_pred = plt.subplots()
        ax_pred.bar(['Extrovert', 'Introvert'], pred_counts)
        ax_pred.set_ylabel("Jumlah")
        ax_pred.set_title("Distribusi Prediksi")
        st.pyplot(fig_pred)

        # Evaluasi jika ada label
        if 'Personality' in df.columns:
            y_true = LabelEncoder().fit_transform(df['Personality'])
            y_pred = predictions

            st.subheader("🧪 Evaluasi Model")
            st.text("Classification Report:\n" + classification_report(y_true, y_pred))

            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

        # Unduh hasil prediksi
        st.subheader("⬇️ Unduh Hasil Prediksi")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "hasil_prediksi.csv", "text/csv")

    else:
        st.error(f"❌ Kolom fitur tidak lengkap. Harus ada: {feature_columns}")
else:
    # Tampilkan informasi jika belum ada file yang diunggah
    st.info("👆 Silakan unggah file CSV di sidebar untuk memulai prediksi kepribadian.")