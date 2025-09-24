import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Function to load the trained model
@st.cache_data  # caching so it doesn't reload every interaction
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    st.title("Student Score Prediction App")
    st.write("Enter number of hours studied to predict the exam score")

    model = load_model()

    # Input: hours studied
    hours = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=5.0, step=0.1)

    if st.button("Predict Score"):
        # Prepare input for model (reshape if needed)
        input_data = np.array(hours).reshape(-1, 1)
        prediction = model.predict(input_data)
        st.success(f"Predicted Score: {prediction[0]:.2f}")

    st.write("---")

if __name__ == '__main__':
    main()
