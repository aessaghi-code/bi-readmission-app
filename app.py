import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Patient Readmission Risk Profile", layout="wide")

st.title("Patient Readmission Risk Profile")
st.write(
    "This tool helps healthcare organizations identify patients at higher risk of hospital readmission using a reusable composite risk score."
)

# STUDENT NOTE: Define the required columns needed to compute the metric from any uploaded healthcare dataset
REQUIRED_COLUMNS = [
    "encounter_id",
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "number_diagnoses",
    "number_inpatient",
    "readmitted"
]

# STUDENT NOTE: Create a mapping from age groups to numeric midpoint values so age can be included in the weighted score
AGE_MAP = {
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

# STUDENT NOTE: Define a normalization function to place all score inputs on the same 0 to 1 scale
def normalize(series):
    if series.max() == series.min():
        return pd.Series([0] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())

# STUDENT NOTE: Convert readmission labels into a binary flag for validation analysis
def convert_readmitted(value):
    if value in ["<30", ">30"]:
        return 1
    return 0

# STUDENT NOTE: Compute the full metric step by step so the uploaded dataset is scored consistently inside the app
def compute_metric(df):
    result_df = df.copy()

    # STUDENT NOTE: Convert age categories into numeric midpoint values for use in the score
    result_df["age_numeric"] = result_df["age"].map(AGE_MAP)

    # STUDENT NOTE: Remove rows where age could not be mapped because the score requires a numeric age input
    result_df = result_df.dropna(subset=["age_numeric"]).copy()

    # STUDENT NOTE: Convert readmission outcome into a binary validation flag
    result_df["readmitted_flag"] = result_df["readmitted"].apply(convert_readmitted)

    # STUDENT NOTE: Normalize each score input separately so no variable dominates because of scale alone
    result_df["inpatient_norm"] = normalize(result_df["number_inpatient"])
    result_df["diagnoses_norm"] = normalize(result_df["number_diagnoses"])
    result_df["medications_norm"] = normalize(result_df["num_medications"])
    result_df["hospital_norm"] = normalize(result_df["time_in_hospital"])
    result_df["age_norm"] = normalize(result_df["age_numeric"])

    # STUDENT NOTE: Compute the weighted composite readmission risk score using the same logic as the notebook
    result_df["risk_score"] = (
        0.30 * result_df["inpatient_norm"] +
        0.20 * result_df["diagnoses_norm"] +
        0.20 * result_df["medications_norm"] +
        0.15 * result_df["hospital_norm"] +
        0.15 * result_df["age_norm"]
    )

    # STUDENT NOTE: Use quantile thresholds so the app creates balanced low, medium, and high risk tiers
    q1 = result_df["risk_score"].quantile(0.33)
    q2 = result_df["risk_score"].quantile(0.66)

    def assign_tier(score):
        if score <= q1:
            return "Low"
        elif score <= q2:
            return "Medium"
        else:
            return "High"

    result_df["risk_tier"] = result_df["risk_score"].apply(assign_tier)

    return result_df


uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:

    # STUDENT NOTE: Load the uploaded dataset into a DataFrame for validation and scoring
    df = pd.read_csv(uploaded_file)

    # STUDENT NOTE: Validate that all required columns are present before any analysis is attempted
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}. Please check your file.")
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head(10))

    # STUDENT NOTE: Add an interactive filter based on minimum hospital stay so the analysis updates on a meaningful patient subset
    min_stay = st.slider(
        "Minimum time in hospital (days)",
        min_value=int(df["time_in_hospital"].min()),
        max_value=int(df["time_in_hospital"].max()),
        value=int(df["time_in_hospital"].min())
    )

    # STUDENT NOTE: Filter the dataset before computing the metric so the slider changes the output values
    filtered_df = df[df["time_in_hospital"] >= min_stay].copy()

    if filtered_df.empty:
        st.warning("No records match the current filter. Please lower the minimum hospital stay.")
        st.stop()

    # STUDENT NOTE: Compute the risk score and tiers on the filtered dataset
    result_df = compute_metric(filtered_df)

    if result_df.empty:
        st.error("The uploaded file could not be scored after age mapping. Please check the age format in the dataset.")
        st.stop()

    # STUDENT NOTE: Calculate KPI values for headline presentation
    high_risk_pct = (result_df["risk_tier"] == "High").mean() * 100
    avg_risk_score = result_df["risk_score"].mean()
    high_risk_validation_rate = result_df[result_df["risk_tier"] == "High"]["readmitted_flag"].mean() * 100

    st.subheader("Headline Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("High-Risk Patients (%)", f"{high_risk_pct:.1f}%")

    with col2:
        st.metric("Average Risk Score", f"{avg_risk_score:.3f}")

    with col3:
        st.metric("High-Risk Readmission Rate", f"{high_risk_validation_rate:.1f}%")

    st.subheader("Risk Tier Distribution")

    # STUDENT NOTE: Count patients in each risk tier for donut chart display
    tier_counts = result_df["risk_tier"].value_counts().reset_index()
    tier_counts.columns = ["risk_tier", "count"]

    fig_donut = px.pie(
        tier_counts,
        names="risk_tier",
        values="count",
        hole=0.4,
        title="Risk Tier Distribution"
    )

    st.plotly_chart(fig_donut, use_container_width=True)

    st.subheader("Risk Score Distribution")

    # STUDENT NOTE: Create an interactive histogram of the computed risk scores
    fig_hist = px.histogram(
        result_df,
        x="risk_score",
        nbins=30,
        title="Distribution of Readmission Risk Scores"
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("High-Risk Patient Profile")

    # STUDENT NOTE: Isolate high-risk patients and summarize their average normalized factor values
    high_risk = result_df[result_df["risk_tier"] == "High"]

    profile_df = pd.DataFrame({
        "Risk Factor": [
            "Prior inpatient visits",
            "Diagnoses",
            "Medications",
            "Hospital stay",
            "Age"
        ],
        "Average normalized value": [
            high_risk["inpatient_norm"].mean(),
            high_risk["diagnoses_norm"].mean(),
            high_risk["medications_norm"].mean(),
            high_risk["hospital_norm"].mean(),
            high_risk["age_norm"].mean()
        ]
    })

    fig_profile = px.bar(
        profile_df,
        x="Risk Factor",
        y="Average normalized value",
        title="High Risk Patient Profile"
    )

    st.plotly_chart(fig_profile, use_container_width=True)

    st.subheader("Interpretation")

    st.info(
        "This tool estimates which patients are most likely to return for a hospital readmission "
        "based on prior inpatient history, number of diagnoses, medication burden, time in hospital, "
        "and age. In the current filtered dataset, the highest-risk tier is mainly associated with "
        "older patients, longer hospital stays, and greater clinical complexity. Healthcare managers "
        "should pay special attention to this group when designing discharge planning, follow-up, "
        "and care coordination interventions."
    )
