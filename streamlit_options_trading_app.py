
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("sample_options_data.csv")
X = df[["underlying_price", "strike_price", "option_price", "iv", "dte", "volume"]]
y = df["profit"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("AI Options Trading Predictor")
st.write("Predict whether a call option trade will be profitable or not.")

underlying_price = st.slider("Underlying Price", 90.0, 130.0, 110.0)
strike_price = st.selectbox("Strike Price", [100, 105, 110, 115, 120])
option_price = st.slider("Option Price", 1.0, 10.0, 2.7)
iv = st.slider("Implied Volatility (IV)", 0.1, 0.5, 0.23)
dte = st.slider("Days to Expiry (DTE)", 1, 60, 10)
volume = st.slider("Volume", 100, 2000, 350)

input_data = pd.DataFrame([{
    "underlying_price": underlying_price,
    "strike_price": strike_price,
    "option_price": option_price,
    "iv": iv,
    "dte": dte,
    "volume": volume
}])

if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "✅ Profitable Trade" if prediction[0] == 1 else "❌ Not Profitable"
    st.subheader(f"Prediction: {result}")

# Feature Importance Plot
st.subheader("Feature Importance")
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance")

st.bar_chart(importance_df.set_index("Feature"))
