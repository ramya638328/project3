import streamlit as st
import pickle
import numpy as np
import os

st.title("🎓 Student Grade Prediction App")

# Debug: show files in directory
st.write("Files in directory:", os.listdir())

MODEL_PATH = "StudentMarkDataset.pkl"

# Check file existence
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found ❌")
    st.stop()

# Load model + encoders
with open(MODEL_PATH, "rb") as f:
    model, le_branch, le_course, le_grade = pickle.load(f)

# Inputs
branch = st.selectbox("Select Branch", le_branch.classes_)
course = st.selectbox("Select Course", le_course.classes_)
marks = st.slider("Enter Marks", 0, 100, 75)

# Encode inputs
branch_encoded = le_branch.transform([branch])[0]
course_encoded = le_course.transform([course])[0]

# Predict
if st.button("Predict Grade"):
    input_data = np.array([[branch_encoded, course_encoded, marks]])
    prediction = model.predict(input_data)
    grade = le_grade.inverse_transform(prediction)
    st.success(f"Predicted Grade: {grade[0]}")
