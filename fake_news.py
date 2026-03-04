import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("news.csv")

# Get labels
labels = df["label"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    df["text"], labels, test_size=0.2, random_state=7
)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)

tfidf_train = tfidf.fit_transform(x_train)
tfidf_test = tfidf.transform(x_test)

# Passive Aggressive Classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict
y_pred = pac.predict(tfidf_test)

# Accuracy
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(score*100, 2)}%")

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"]))

# Test with your own input
while True:
    news = input("\nEnter a news headline (or type 'quit' to exit): ")

    if news.lower() == "quit":
        break

    news_vector = tfidf.transform([news])
    prediction = pac.predict(news_vector)

    print("Prediction:", prediction[0])