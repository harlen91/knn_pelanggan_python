import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Kategori Pelanggan", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

st.title("Prediksi Kategori pelanggan")
st.write("Masukkan data pelanggan untuk memprediksi menggunakan model KNN")
st.markdown("----")

with st.form("prediction_form"):
    st.subheader("Data Pelanggan")

    income = st.number_input("Pendapatan Tahunan", min_value=1, max_value=500, value=50)
    spending_score = st.slider("Skor Pengeluaran", min_value=1, max_value=100, value=50)
    age = st.number_input("Usia Pelanggan", min_value=10, max_value=100, value=30)

    submitted = st.form_submit_button("Prediksi Kategori")

    if submitted:
        input_df = pd.DataFrame(
            [[income,spending_score,age]], columns=['Income_Annual_k$','Spending_Score','Age']
        )

        input_scaled = scaler.transform(input_df)
        prediksi = model.predict(input_scaled)[0]

        st.success(f"Hasil Prediksi : **{prediksi}**")

        st.markdown('** Ringakasa Nilai Input : **')
        st.write(input_df)