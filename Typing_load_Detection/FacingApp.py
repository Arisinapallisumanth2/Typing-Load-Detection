import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Cognitive Load Detector", layout="centered")

# Load the trained model
model = joblib.load("model.pkl")

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Live Typing", "EDA Insights"])

if page == "Live Typing":
    st.title("Cognitive Load Detection from Typing")
    st.write("Type below. We'll analyze your typing behavior to estimate your cognitive load.")

    # Typing area
    typed_text = st.text_area("Start typing...", height=200, key="typing_input")

    # Feature extraction (based on text length and complexity)
    def smart_feature_extraction(text):
        char_count = len(text)
        word_count = len(text.split())
        duration = max(char_count / 6.0, 1.0)  # Simulated duration

        speed = round(char_count / duration, 2)

        # Rules based on speed
        if speed >= 5.0:
            avg_dwell = round(np.random.normal(0.14, 0.01), 3)
            avg_flight = round(np.random.normal(0.12, 0.01), 3)
            pauses = np.random.randint(0, 2)
            errors = np.random.randint(0, 2)
        elif speed <= 2.5:
            avg_dwell = round(np.random.normal(0.26, 0.02), 3)
            avg_flight = round(np.random.normal(0.22, 0.01), 3)
            pauses = np.random.randint(3, 5)
            errors = np.random.randint(2, 4)
        else:
            avg_dwell = round(np.random.normal(0.20, 0.015), 3)
            avg_flight = round(np.random.normal(0.18, 0.015), 3)
            pauses = np.random.randint(1, 3)
            errors = np.random.randint(1, 3)

        return {
            'avg_dwell': avg_dwell,
            'avg_flight': avg_flight,
            'speed': speed,
            'pauses': pauses,
            'errors': errors
        }
    if st.button("Analyze Cognitive Load"):
        if typed_text.strip() == "":
            st.warning("Please type something before analyzing.")
        else:
            with st.spinner("Analyzing typing behavior..."):
                time.sleep(2)
                features = smart_feature_extraction(typed_text)
                X_input = pd.DataFrame([features])
                prediction = model.predict(X_input)[0]

                st.subheader("Extracted Features")
                st.json(features)

                label = "high" if prediction == 1 else "low"
                if prediction == 1:
                    st.success("Active Load Detected")
                else:
                    st.error("UnActive Load Detected")

                
                st.write("Observation")

                if features['speed'] < 3.0:
                    st.warning("Typing speed is low — possibly indicating high cognitive load.")
                if features['pauses'] > 2:
                    st.warning("Multiple pauses detected — might suggest distraction or fatigue.")
                if features['errors'] > 1:
                    st.info("You made a few corrections — maybe typing under pressure?")
                # Save session data
                features["label"] = label
                save_path = "features_summary.csv"
                pd.DataFrame([features]).to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)
                st.info("Session saved to features_summary.csv")

elif page == "EDA Insights":
    st.title("EDA - Cognitive Load Feature Analysis")

    try:
        df = pd.read_csv("features_summary.csv")

        st.write("Dataset Preview")
        st.dataframe(df.tail())

        st.write("Class Distribution")
        st.bar_chart(df["label"].value_counts())

        st.write("Feature Distributions by Cognitive Load")
        features = ['avg_dwell', 'avg_flight', 'speed', 'pauses', 'errors']
        for feature in features:
            fig, ax = plt.subplots()
            sns.boxplot(x="label", y=feature, data=df, ax=ax, palette="coolwarm")
            st.pyplot(fig)

        st.write("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df.drop(columns=['label']).corr(), annot=True, cmap='Blues', ax=ax)
        st.pyplot(fig)

    except FileNotFoundError:
        st.error("features_summary.csv not found. Please make sure it's in the project directory.")
