# ===========================================================================
#  AI Student Burnout Risk Predictor — Streamlit Web App
#  Final Project AI & Big Data 2026
# ---------------------------------------------------------------------------
#  Aplikasi ini memuat model Random Forest yang sudah dilatih di Google Colab,
#  menerima input dari pengguna (mahasiswa), lalu memprediksi apakah mahasiswa
#  tersebut berisiko mengalami BURNOUT TINGGI.
# ===========================================================================

import streamlit as st          # framework untuk membuat web app dengan Python
import pandas as pd             # untuk membuat DataFrame input
import numpy as np              # operasi numerik
import joblib                   # untuk memuat file model .pkl

# ---------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Burnout Risk Predictor",   # judul tab browser
    page_icon="🎓",                            # ikon tab
    layout="centered",                          # tata letak di tengah
)

# ---------------------------------------------------------------------------
# 2. MUAT MODEL (di-cache agar tidak dimuat ulang setiap interaksi)
# ---------------------------------------------------------------------------
@st.cache_resource   # cache: model hanya dimuat sekali, mempercepat aplikasi
def load_artifacts():
    """Memuat model, scaler, daftar fitur, dan info akurasi dari file .pkl."""
    model = joblib.load("model.pkl")                 # model Random Forest terbaik
    scaler = joblib.load("scaler.pkl")               # scaler untuk standarisasi
    feature_columns = joblib.load("feature_columns.pkl")  # urutan 23 kolom fitur
    model_info = joblib.load("model_info.pkl")       # nama model + akurasi
    return model, scaler, feature_columns, model_info

model, scaler, feature_columns, model_info = load_artifacts()

# ---------------------------------------------------------------------------
# 3. HEADER / JUDUL APLIKASI
# ---------------------------------------------------------------------------
st.title("🎓 AI Student Burnout Risk Predictor")
st.markdown(
    "Prediksi **risiko burnout** mahasiswa berdasarkan pola penggunaan "
    "AI generatif dan kebiasaan belajar. *Final Project AI & Big Data 2026.*"
)

# Tampilkan akurasi model (WAJIB ada di website sesuai ketentuan tugas)
col_a, col_b = st.columns(2)
col_a.metric("🏆 Model Terbaik", model_info["nama"])
col_b.metric("🎯 Akurasi Model", f"{model_info['akurasi']*100:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# 4. FORM INPUT PENGGUNA  (lebih dari 4 fitur, sesuai ketentuan)
# ---------------------------------------------------------------------------
st.subheader("📝 Masukkan Data Mahasiswa")

# --- Kolom kiri & kanan agar form rapi ---
c1, c2 = st.columns(2)

with c1:
    # FITUR 1: jam penggunaan AI per minggu (prediktor terkuat)
    weekly_ai_hours = st.slider(
        "Jam penggunaan AI per minggu", 0.0, 40.0, 8.0, 0.5,
        help="Total jam memakai ChatGPT/Gemini/Copilot dll per minggu"
    )
    # FITUR 2: jam belajar tradisional per minggu
    study_hours = st.slider(
        "Jam belajar tradisional per minggu", 0.0, 36.0, 11.0, 0.5,
        help="Jam belajar mandiri tanpa bantuan AI"
    )
    # FITUR 3: tingkat ketergantungan pada AI
    ai_dependency = st.slider(
        "Tingkat ketergantungan pada AI (1–10)", 1, 10, 4,
        help="1 = tidak bergantung, 10 = sangat bergantung"
    )
    # FITUR 4: tingkat kecemasan saat ujian
    anxiety = st.slider(
        "Tingkat kecemasan saat ujian (1–10)", 1, 10, 4
    )
    # FITUR 5: skor retensi keterampilan
    skill_retention = st.slider(
        "Skor retensi keterampilan (0–100)", 0, 100, 75
    )

with c2:
    # FITUR 6: IPK sebelum semester
    pre_gpa = st.slider("IPK sebelum semester", 1.0, 4.0, 3.1, 0.01)
    # FITUR 7: IPK sesudah semester (untuk fitur turunan GPA_Improved)
    post_gpa = st.slider("IPK sesudah semester", 1.0, 4.0, 3.3, 0.01)
    # FITUR 8: jumlah jenis tool AI
    tool_diversity = st.slider("Jumlah jenis tool AI dipakai", 1, 5, 3)
    # FITUR 9: keterampilan prompt engineering
    prompt_skill = st.selectbox(
        "Keterampilan prompt engineering",
        ["Beginner", "Intermediate", "Advanced"]
    )
    # FITUR 10: tingkat studi
    year = st.selectbox(
        "Tingkat studi",
        ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]
    )

# Baris kedua untuk fitur kategorikal lainnya
c3, c4, c5 = st.columns(3)
with c3:
    # FITUR 11: langganan berbayar
    paid = st.selectbox("Langganan AI berbayar?", ["Tidak", "Ya"])
with c4:
    # FITUR 12: rumpun jurusan
    major = st.selectbox(
        "Rumpun jurusan",
        ["Arts", "Business", "Humanities", "Medical", "STEM"]
    )
with c5:
    # FITUR 13: tujuan utama pakai AI
    use_case = st.selectbox(
        "Tujuan utama memakai AI",
        ["Copywriting/Drafting", "Debugging/Troubleshooting",
         "Direct_Answer_Generation", "Ideation", "Summarizing_Reading"]
    )

# Kebijakan kampus
policy = st.selectbox(
    "Kebijakan kampus soal AI",
    ["Actively_Encouraged", "Allowed_With_Citation", "Strict_Ban"]
)

st.divider()

# ---------------------------------------------------------------------------
# 5. FUNGSI MEMBANGUN VEKTOR FITUR
#    (harus PERSIS sama dengan proses di notebook training)
# ---------------------------------------------------------------------------
def build_feature_vector():
    """Mengubah input pengguna menjadi 1 baris DataFrame dengan 23 kolom
    yang urutannya sama persis dengan saat model dilatih."""

    # Mulai dengan DataFrame berisi nol untuk semua kolom fitur
    row = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    # --- Encoding ORDINAL (sama seperti di notebook) ---
    skill_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    year_map = {"Freshman": 0, "Sophomore": 1, "Junior": 2, "Senior": 3, "Graduate": 4}

    # --- Isi fitur numerik & ordinal ---
    row["Year_of_Study"] = year_map[year]
    row["Pre_Semester_GPA"] = pre_gpa
    row["Weekly_GenAI_Hours"] = weekly_ai_hours
    row["Prompt_Engineering_Skill"] = skill_map[prompt_skill]
    row["Tool_Diversity"] = tool_diversity
    row["Paid_Subscription"] = 1 if paid == "Ya" else 0
    row["Traditional_Study_Hours"] = study_hours
    row["Perceived_AI_Dependency"] = ai_dependency
    row["Anxiety_Level_During_Exams"] = anxiety
    row["Skill_Retention_Score"] = skill_retention

    # --- Fitur hasil FEATURE ENGINEERING (rumus sama dengan notebook) ---
    row["AI_to_Study_Ratio"] = weekly_ai_hours / (study_hours + 1)
    row["Total_Study_Load"] = weekly_ai_hours + study_hours
    row["GPA_Improved"] = 1 if post_gpa > pre_gpa else 0

    # --- One-Hot Encoding (drop_first=True, jadi kategori pertama = semua 0) ---
    # Major_Category: kategori dasar 'Arts' (tidak punya kolom)
    major_col = f"Major_Category_{major}"
    if major_col in row.columns:
        row[major_col] = 1

    # Primary_Use_Case: kategori dasar 'Copywriting/Drafting'
    use_col = f"Primary_Use_Case_{use_case}"
    if use_col in row.columns:
        row[use_col] = 1

    # Institutional_Policy: kategori dasar 'Actively_Encouraged'
    policy_col = f"Institutional_Policy_{policy}"
    if policy_col in row.columns:
        row[policy_col] = 1

    return row

# ---------------------------------------------------------------------------
# 6. TOMBOL PREDIKSI
# ---------------------------------------------------------------------------
if st.button("🔮 Prediksi Risiko Burnout", type="primary", use_container_width=True):

    # Bangun vektor fitur dari input
    X_input = build_feature_vector()

    # Standarisasi memakai scaler yang sama seperti saat training
    X_scaled = scaler.transform(X_input)

    # Lakukan prediksi
    pred = model.predict(X_scaled)[0]                  # 0 atau 1
    proba = model.predict_proba(X_scaled)[0][1]        # probabilitas kelas High

    st.divider()
    st.subheader("📊 Hasil Prediksi")

    # Tampilkan hasil dengan warna sesuai tingkat risiko
    if pred == 1:
        st.error(f"⚠️ **RISIKO BURNOUT TINGGI**")
        st.markdown(
            f"Model memperkirakan mahasiswa ini **berisiko tinggi** mengalami burnout "
            f"dengan probabilitas **{proba*100:.1f}%**."
        )
        st.markdown(
            "**Rekomendasi:** Pertimbangkan untuk mengurangi jam penggunaan AI, "
            "menambah waktu istirahat, dan berkonsultasi dengan dosen pembimbing "
            "atau layanan konseling kampus."
        )
    else:
        st.success(f"✅ **RISIKO BURNOUT RENDAH / SEDANG**")
        st.markdown(
            f"Model memperkirakan mahasiswa ini **tidak** berisiko burnout tinggi "
            f"(probabilitas burnout tinggi hanya **{proba*100:.1f}%**)."
        )
        st.markdown(
            "**Rekomendasi:** Pertahankan keseimbangan antara penggunaan AI dan "
            "belajar mandiri yang sudah baik."
        )

    # Progress bar untuk visualisasi probabilitas
    st.markdown("**Probabilitas Burnout Tinggi:**")
    st.progress(float(proba))

# ---------------------------------------------------------------------------
# 7. FOOTER
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    f"Model: {model_info['nama']} • Akurasi: {model_info['akurasi']*100:.1f}% • "
    f"F1-Score: {model_info['f1']:.3f} • Dataset: 50.000 mahasiswa"
)
