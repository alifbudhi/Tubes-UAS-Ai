
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="AI Burnout Risk Predictor Kelompok 3",
    page_icon="🍌",
    layout="centered",
)

# ===== BACKGROUND RAINBOW KELAP-KELIP + EMOJI BERGERAK =====
st.markdown("""
<style>
/* Latar belakang rainbow yang bergerak cepat (lebih meriah) */
.stApp {
    background: linear-gradient(-45deg,
        #ff5f6d, #ffc371, #fff75e, #44ff9a, #44b0ff, #8b5cff, #ff5fd2, #ff5f6d);
    background-size: 400% 400%;
    animation: rainbowShift 6s linear infinite;
}
@keyframes rainbowShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
/* Lapisan kelap-kelip (twinkle) di atas background */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(2px 2px at 20% 30%, #fff, transparent),
        radial-gradient(2px 2px at 60% 70%, #fff, transparent),
        radial-gradient(3px 3px at 50% 20%, #fff, transparent),
        radial-gradient(2px 2px at 80% 40%, #fff, transparent),
        radial-gradient(2px 2px at 35% 80%, #fff, transparent),
        radial-gradient(3px 3px at 90% 85%, #fff, transparent),
        radial-gradient(2px 2px at 10% 60%, #fff, transparent);
    background-size: 200% 200%;
    animation: twinkle 2.2s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
@keyframes twinkle {
    0%, 100% { opacity: 0.15; }
    50%      { opacity: 0.9; }
}
/* Panel gelap semi-transparan di belakang konten agar teks tetap terbaca */
.stApp .block-container {
    background: rgba(15, 23, 42, 0.82);
    padding: 2rem 2.5rem;
    border-radius: 18px;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    z-index: 1;
    box-shadow: 0 0 40px rgba(255, 255, 255, 0.25);
}
/* Pastikan teks tetap terang di atas panel gelap */
.stApp .block-container,
.stApp .block-container p,
.stApp .block-container label,
.stApp .block-container h1,
.stApp .block-container h2,
.stApp .block-container h3,
.stApp .block-container li {
    color: #f1f5f9 !important;
}
/* Emoji yang memantul (bouncing) */
.emoji-bounce {
    display: inline-block;
    animation: bounce 1.0s ease infinite;
    font-size: 2.4rem;
}
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-16px); }
}
/* Emoji yang bergoyang (wiggle) */
.emoji-wiggle {
    display: inline-block;
    animation: wiggle 1.2s ease-in-out infinite;
    font-size: 2.4rem;
}
@keyframes wiggle {
    0%, 100% { transform: rotate(-18deg); }
    50%      { transform: rotate(18deg); }
}
/* Emoji pisang yang berputar pelan */
.emoji-spin {
    display: inline-block;
    animation: spin 2.5s linear infinite;
    font-size: 2.2rem;
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Memuat model, scaler, daftar fitur, dan info akurasi dari file .pkl."""
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    model_info = joblib.load("model_info.pkl")
    return model, scaler, feature_columns, model_info
model, scaler, feature_columns, model_info = load_artifacts()


# Judul dengan emoji bergerak (pisang berputar + memantul + bergoyang)
st.markdown(
    '<h1>'
    '<span class="emoji-spin">🍌</span> '
    '<span class="emoji-bounce">🎓</span> AI Student Burnout Risk Predictor '
    '<span class="emoji-wiggle">🌈</span> '
    '<span class="emoji-spin">🍌</span>'
    '</h1>',
    unsafe_allow_html=True
)
st.markdown(
    "Prediksi **risiko burnout** mahasiswa berdasarkan pola penggunaan "
    "AI generatif dan kebiasaan belajar. Final Project AI & Big Data 2026 Kelompok 3."
)

col_a, col_b = st.columns(2)
col_a.metric("🍌 Model Terbaik", model_info["nama"])
col_b.metric("🍌 Akurasi Model", f"{model_info['akurasi']*100:.1f}%")

st.divider()

st.subheader("🍌 Masukkan Data Mahasiswa")

c1, c2 = st.columns(2)

with c1:
    weekly_ai_hours = st.slider(
        "Jam penggunaan AI per minggu", 0.0, 40.0, 8.0, 0.5,
        help="Total jam memakai ChatGPT/Gemini/Copilot dll per minggu"
    )
    study_hours = st.slider(
        "Jam belajar tradisional per minggu", 0.0, 36.0, 11.0, 0.5,
        help="Jam belajar mandiri tanpa bantuan AI"
    )
    ai_dependency = st.slider(
        "Tingkat ketergantungan pada AI (1–10)", 1, 10, 4,
        help="1 = tidak bergantung, 10 = sangat bergantung"
    )
    anxiety = st.slider(
        "Tingkat kecemasan saat ujian (1–10)", 1, 10, 4
    )
    skill_retention = st.slider(
        "Skor retensi keterampilan (0–100)", 0, 100, 75
    )

with c2:
    pre_gpa = st.slider("IPK sebelum semester", 1.0, 4.0, 3.1, 0.01)
    post_gpa = st.slider("IPK sesudah semester", 1.0, 4.0, 3.3, 0.01)
    tool_diversity = st.slider("Jumlah jenis tool AI dipakai", 1, 5, 3)
    prompt_skill = st.selectbox(
        "Keterampilan prompt engineering",
        ["Beginner", "Intermediate", "Advanced"]
    )
    year = st.selectbox(
        "Tingkat studi",
        ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]
    )
c3, c4, c5 = st.columns(3)
with c3:
    paid = st.selectbox("Langganan AI berbayar?", ["Tidak", "Ya"])
with c4:
    major = st.selectbox(
        "Rumpun jurusan",
        ["Arts", "Business", "Humanities", "Medical", "STEM"]
    )
with c5:
    use_case = st.selectbox(
        "Tujuan utama memakai AI",
        ["Copywriting/Drafting", "Debugging/Troubleshooting",
         "Direct_Answer_Generation", "Ideation", "Summarizing_Reading"]
    )

policy = st.selectbox(
    "Kebijakan kampus soal AI",
    ["Actively_Encouraged", "Allowed_With_Citation", "Strict_Ban"]
)

st.divider()

def build_feature_vector():
    """Mengubah input pengguna menjadi 1 baris DataFrame."""

    row = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    skill_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    year_map = {"Freshman": 0, "Sophomore": 1, "Junior": 2, "Senior": 3, "Graduate": 4}

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

    row["AI_to_Study_Ratio"] = weekly_ai_hours / (study_hours + 1)
    row["Total_Study_Load"] = weekly_ai_hours + study_hours
    row["GPA_Improved"] = 1 if post_gpa > pre_gpa else 0

    major_col = f"Major_Category_{major}"
    if major_col in row.columns:
        row[major_col] = 1

    use_col = f"Primary_Use_Case_{use_case}"
    if use_col in row.columns:
        row[use_col] = 1

    policy_col = f"Institutional_Policy_{policy}"
    if policy_col in row.columns:
        row[policy_col] = 1

    return row


if st.button("🍌 Prediksi Risiko Burnout", type="primary", use_container_width=True):

    X_input = build_feature_vector()

    X_scaled = scaler.transform(X_input)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]

    st.divider()
    st.subheader("🍌 Hasil Prediksi")

    if pred == 1:
        st.error("🍌🍌 **RISIKO BURNOUT TINGGI**")
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
        st.success("✅ **RISIKO BURNOUT RENDAH / SEDANG**")
        st.markdown(
            f"Model memperkirakan mahasiswa ini **tidak** berisiko burnout tinggi "
            f"(probabilitas burnout tinggi hanya **{proba*100:.1f}%**)."
        )
        st.markdown(
            "**Rekomendasi:** Pertahankan keseimbangan antara penggunaan AI dan "
            "belajar mandiri yang sudah baik."
        )

    st.markdown("**Probabilitas Burnout Tinggi:**")
    st.progress(float(proba))

st.divider()
st.caption(
    f"Model: {model_info['nama']} • Akurasi: {model_info['akurasi']*100:.1f}% • "
    f"F1-Score: {model_info['f1']:.3f} • Dataset: 50.000 mahasiswa"
)
