
import streamlit as st          
import pandas as pd             
import numpy as np              
import joblib        

st.set_page_config(
    page_title="AI Burnout Risk Predictor Kelompok 3",   
    page_icon="🍌",                            
    layout="centered",                          
)

@st.cache_resource 
def load_artifacts():
    """Memuat model, scaler, daftar fitur, dan info akurasi dari file .pkl."""
    model = joblib.load("model.pkl")                 
    scaler = joblib.load("scaler.pkl")              
    feature_columns = joblib.load("feature_columns.pkl")  
    model_info = joblib.load("model_info.pkl")       
    return model, scaler, feature_columns, model_info
model, scaler, feature_columns, model_info = load_artifacts()


st.title("🍌 AI Student Burnout Risk Predictor")
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
        ["1", "2", "3", "4", "Graduate"]
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
        st.error(f"🍌🍌 **RISIKO BURNOUT TINGGI**")
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

    st.markdown("**Probabilitas Burnout Tinggi:**")
    st.progress(float(proba))

st.divider()
st.caption(
    f"Model: {model_info['nama']} • Akurasi: {model_info['akurasi']*100:.1f}% • "
    f"F1-Score: {model_info['f1']:.3f} • Dataset: 50.000 mahasiswa"
)
