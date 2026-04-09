import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("data.csv")

X = data["resume"]
y = data["category"]

vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vector, y)

# UI
st.title("AI Resume Screening System")

user_input = st.text_area("Enter Resume Text")

if st.button("Predict"):
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)
    score = model.predict_proba(input_vector).max() * 100

    st.write("Predicted Job Role:", prediction[0])
    st.write("Match Score:", round(score, 2), "%")