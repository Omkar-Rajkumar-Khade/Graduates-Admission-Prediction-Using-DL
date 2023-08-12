import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('models/Graduates Admission Prediction.h5')

# Load the original dataset to initialize the scaler
original_df = pd.read_csv('dataset/admission_data.csv')
scaler = MinMaxScaler()
scaler.fit(original_df.iloc[:, :-1])  # Use all columns except the target

# Streamlit app
st.title("Graduates Admission Prediction")
st.sidebar.header("User Input")

# Sidebar inputs
gre = st.sidebar.number_input("GRE Score", value=320)
toefl = st.sidebar.number_input("TOEFL Score", value=110)
university_rating = st.sidebar.number_input("University Rating", value=4)
sop = st.sidebar.number_input("SOP", value=4.5)
lor = st.sidebar.number_input("LOR", value=4.5)
cgpa = st.sidebar.number_input("CGPA", value=9.0)
research = st.sidebar.number_input("Research (0 for No, 1 for Yes)", value=1)

# Prepare user input
user_input = np.array([gre, toefl, university_rating, sop, lor, cgpa, research]).reshape(1, -1)
user_input_scaled = scaler.transform(user_input)

# Make prediction
prediction = model.predict(user_input_scaled)

# Display prediction
st.write("Predicted Chance of Admit (in percentage %):")
st.write(f"<span style='font-size: 28px;'>{prediction[0][0] * 100:.2f}%</span>", unsafe_allow_html=True)
