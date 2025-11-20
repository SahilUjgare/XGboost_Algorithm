import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Load Saved XGBoost Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("best_XGB_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚀 XGBoost Regression Model – Prediction App")
st.write("Upload your CSV file or manually enter values to get predictions.")

# Tabs for UI
tab1, tab2 = st.tabs(["📂 Upload CSV", "✍️ Manual Input"])

# -----------------------------
# 1️⃣ Upload CSV Prediction
# -----------------------------
with tab1:
    st.subheader("Upload a CSV file for batch prediction")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview:", df.head())

        try:
            predictions = model.predict(df)
            df["Predictions"] = predictions

            st.success("Prediction completed!")
            st.write(df)

            # Download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------
# 2️⃣ Manual Input Form
# -----------------------------
with tab2:
    st.subheader("Enter values manually")

    st.info("Note: Use the same feature names and order as used during training.")

    # Add your model’s feature names here
    # Example placeholder:
    # feature_1 = st.number_input("Feature 1", value=0.0)
    # feature_2 = st.number_input("Feature 2", value=0.0)

    st.warning("⚠️ Replace this section with your actual feature inputs.")

    # Placeholder for demo:
    sample_input = st.text_input("Enter comma-separated values (x1,x2,x3...)")

    if st.button("Predict Manually"):
        try:
            values = np.array(sample_input.split(","), dtype=float).reshape(1, -1)
            result = model.predict(values)[0]
            st.success(f"Predicted Value: **{result}**")
        except:
            st.error("Invalid input format. Enter values like: 12, 45, 3.4, 89")

# Footer
st.markdown("---")
st.caption("Developed using Streamlit & XGBoost")

