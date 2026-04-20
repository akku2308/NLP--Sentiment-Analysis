import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

nltk.download('stopwords')

# Load dataset (IMPORTANT: path sahi rakho)
data = pd.read_csv("nlp_sentiment_large.csv")

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Clean data
data['Clean_Review'] = data['Review'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['Clean_Review'])
y = data['Sentiment']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# UI
st.title("💬 Sentiment Analysis App")

user_input = st.text_area("Enter your review:")
if st.button("Predict"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]

    if prediction == 1:
        st.success(f"😊 Positive (Confidence: {proba[1]*100:.2f}%)")
    else:
        st.error(f"😡 Negative (Confidence: {proba[0]*100:.2f}%)")