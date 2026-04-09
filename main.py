print("Program Running...")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("data.csv")

X = data["resume"]
y = data["category"]

vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vector, y)

new_resume = ["python machine learning deep learning"]
new_vector = vectorizer.transform(new_resume)

prediction = model.predict(new_vector)

# Score calculation
score = model.predict_proba(new_vector).max() * 100

print("Predicted Job Role:", prediction[0])
print("Match Score:", round(score, 2), "%")