import streamlit as st
import pickle
import numpy as np

# Load the model and data
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))  # Load the pipeline using pickle
    df = pickle.load(open('df.pkl', 'rb'))      # Load the DataFrame using pickle
except FileNotFoundError:
    st.error("Model or data file not found. Ensure 'pipe.pkl' and 'df.pkl' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading files: {str(e)}")
    st.stop()

st.title("Laptop Price Predictor")

# Input fields
# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
type_ = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop (in kg)', min_value=0.5, max_value=5.0, step=0.1)

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size
screen_size = st.slider('Screen size (in inches)', 10.0, 18.0, 13.0)

# Resolution
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600',
    '2560x1440', '2304x1440'
])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# Storage options
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# Operating System
os = st.selectbox('OS', df['os'].unique())

# Prediction button
if st.button('Predict Price'):
    # Preprocess input data
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI
    try:
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size
    except ValueError:
        st.error("Invalid resolution format.")
        st.stop()

    # Formulate the query
    query = np.array([company, type_, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, -1)

    # Predict price
    try:
        prediction = np.exp(pipe.predict(query)[0])  # Use exponential for log-transformed output
        st.title(f"The predicted price of this configuration is ₹{int(prediction):,}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
