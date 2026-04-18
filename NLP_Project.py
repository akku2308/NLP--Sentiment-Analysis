# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 07:36:49 2026

@author: ashis
"""
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("C:/Users/ashis/Downloads/nlp_sentiment_large.csv")
print(data.head())
data.shape
print(data.columns)
print(data.info())
print(data.isnull().sum())

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return " ".join(words)

data['Review_text'] = data['Review'].apply(clean_text)

import seaborn as sns
data['Sentiment'].value_counts()
sns.countplot(x='Sentiment' , data=data)
plt.show()

data['Review_Length'] = data['Review'].apply(len)
sns.boxplot(x='Sentiment', y='Review_Length',  data=data)
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
data['Clean_Review'] = data['Review'].apply(clean_text)

X = vectorizer.fit_transform(data['Clean_Review'])
y = data['Sentiment']
print(data.columns)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test) 



from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    
    if prediction == 1:
        return "Positive"
    else:
        return "Negative"
print(predict_sentiment("This product is amazing"))
print(predict_sentiment("Worst experience ever"))

print(data['Sentiment'].unique())
