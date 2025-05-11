import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.title("🌱 Soybean Disease Severity Predictor (Weather & Variety Based)")

# Upload section
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Convert potential numeric columns from object
    for col in ['JS 95-60', 'JS93-05', 'PK -472']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter valid varieties
    ignored_cols = ['Year', 'SMW', 'Crop_Growth_Week', 'Location', 'Longitude', 'Latitude',
                    'Max_Temp', 'Min_Temp', 'Max_Humidity', 'Min_Humidity', 'No of Rainy days',
                    'Rainfall', 'Wind_Velocity', 'Disease', 'Gaurav', ' JS 90-41']
    varieties = [col for col in df.columns if col not in ignored_cols]

    selected_variety = st.selectbox("🌿 Select soybean variety", varieties)

    # 🌱 User picks sowing date
    st.subheader("📆 Select Sowing Date and Current Date")
    sowing_date = st.date_input(
        "🌱 Select Sowing Date",
        min_value=date(2021, 6, 1),
        max_value=date(2021, 12, 31),
        value=date(2021, 6, 15)
    )

    current_date = st.date_input(
        "📅 Select Date for Prediction",
        min_value=sowing_date,
        max_value=date(2021, 12, 31),
        value=sowing_date + pd.Timedelta(days=21)
    )

    # Calculate SMW and Crop Growth Week
    smw = current_date.isocalendar()[1]
    crop_growth_week = max(1, ((current_date - sowing_date).days // 7) + 1)

    st.write(f"📅 Selected Date: {current_date.strftime('%Y-%m-%d')}")
    st.write(f"📈 Standard Meteorological Week (SMW): **{smw}**")
    st.write(f"🌱 Crop Growth Week: **{crop_growth_week}** (since sowing date)")

    # Prepare data for modeling
    features = ['Max_Temp', 'Min_Temp', 'Rainfall', 'Max_Humidity',
                'Min_Humidity', 'No of Rainy days', 'Crop_Growth_Week']
    df['Crop_Growth_Week'] = pd.to_numeric(df['Crop_Growth_Week'], errors='coerce')
    df_model = df[features + [selected_variety]].dropna()

    if df_model.empty:
        st.warning("⚠️ Not enough complete data to train model for this variety.")
    else:
        st.subheader("📊 Correlation Heatmap")
        corr = df_model.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax)
        st.pyplot(fig)

        # Train model
        X = df_model[features]
        y = df_model[selected_variety]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)

        st.subheader(f"📈 Regression Results for {selected_variety}")
        st.write(f"**R² Score:** {score:.3f}")
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_
        })
        st.dataframe(coef_df)

        st.subheader(f"🔍 Predict Severity for {selected_variety}")
        input_data = {}
        for feature in features[:-1]:  # All except CGW
            input_data[feature] = st.number_input(f"{feature}", value=float(df[feature].mean()))
        input_data['Crop_Growth_Week'] = crop_growth_week

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            st.success(f"✅ Predicted disease severity for **{selected_variety}**: **{prediction:.2f}**")
