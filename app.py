import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Patient Readmission Risk Profile")

st.write(
    "This tool helps healthcare organizations identify patients at higher risk of hospital readmission using a composite risk score."
)

# Upload file
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # -------------------------
    # 1. AGE MAPPING
    # -------------------------
    age_map = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95
    }

    df["age_numeric"] = df["age"].map(age_map)
    df = df.dropna(subset=["age_numeric"])

    # -------------------------
    # 2. NORMALIZATION
    # -------------------------
    def normalize(column):
        return (column - column.min()) / (column.max() - column.min())

    df["inpatient_norm"] = normalize(df["number_inpatient"])
    df["diagnoses_norm"] = normalize(df["number_diagnoses"])
    df["medications_norm"] = normalize(df["num_medications"])
    df["hospital_norm"] = normalize(df["time_in_hospital"])
    df["age_norm"] = normalize(df["age_numeric"])

    # -------------------------
    # 3. RISK SCORE
    # -------------------------
    df["risk_score"] = (
        0.30 * df["inpatient_norm"] +
        0.20 * df["diagnoses_norm"] +
        0.20 * df["medications_norm"] +
        0.15 * df["hospital_norm"] +
        0.15 * df["age_norm"]
    )

    # -------------------------
    # 4. RISK TIERS
    # -------------------------
    q1 = df["risk_score"].quantile(0.33)
    q2 = df["risk_score"].quantile(0.66)

    def risk_tier(score):
        if score <= q1:
            return "Low"
        elif score <= q2:
            return "Medium"
        else:
            return "High"

    df["risk_tier"] = df["risk_score"].apply(risk_tier)

    # -------------------------
    # 5. HISTOGRAM
    # -------------------------
    st.subheader("Distribution of Risk Scores")

    fig1, ax1 = plt.subplots()
    ax1.hist(df["risk_score"], bins=30)
    ax1.set_xlabel("Risk Score")
    ax1.set_ylabel("Number of Patients")
    st.pyplot(fig1)

    # -------------------------
    # 6. DONUT CHART
    # -------------------------
    st.subheader("Risk Tier Distribution")

    tier_counts = df["risk_tier"].value_counts()

    fig2, ax2 = plt.subplots()
    ax2.pie(
        tier_counts,
        labels=tier_counts.index,
        autopct="%1.1f%%",
        wedgeprops={"width": 0.4}
    )
    st.pyplot(fig2)

    # -------------------------
    # 7. HIGH RISK PROFILE
    # -------------------------
    st.subheader("High Risk Patient Profile")

    high_risk = df[df["risk_tier"] == "High"]

    profile = {
        "Prior inpatient visits": high_risk["inpatient_norm"].mean(),
        "Diagnoses": high_risk["diagnoses_norm"].mean(),
        "Medications": high_risk["medications_norm"].mean(),
        "Hospital stay": high_risk["hospital_norm"].mean(),
        "Age": high_risk["age_norm"].mean()
    }

    names = list(profile.keys())
    values = list(profile.values())

    fig3, ax3 = plt.subplots()
    ax3.bar(names, values)
    ax3.set_ylabel("Average normalized value")
    ax3.set_xticklabels(names, rotation=20)
    st.pyplot(fig3)

    # -------------------------
    # 8. DOWNLOAD DATASET
    # -------------------------
    st.subheader("Download Results")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download dataset with risk scores",
        data=csv,
        file_name="processed_dataset.csv",
        mime="text/csv"
    )
