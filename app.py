
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Page config
st.set_page_config(page_title="AI Resume Screening", layout="centered")

# Custom CSS for colors
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        color: #4CAF50;
        font-size: 40px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 18px;
    }
    .result-box {
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        background-color: #e3f2fd;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🤖 AI Resume Screening System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze resumes and predict job roles using Machine Learning</div>', unsafe_allow_html=True)

st.write("")

# Load data
data = pd.read_csv("data.csv")

X = data["resume"]
y = data["category"]

vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vector, y)

# Input box
user_input = st.text_area("📄 Paste Resume Text Here", height=200)

# Button
if st.button("🔍 Analyze Resume"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter resume text!")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)
        score = model.predict_proba(input_vector).max() * 100

        st.markdown(f"""
        <div class="result-box">
            <h3>✅ Predicted Role: {prediction[0]}</h3>
            <h4>📊 Match Score: {round(score,2)}%</h4>
        </div>
        """, unsafe_allow_html=True)
