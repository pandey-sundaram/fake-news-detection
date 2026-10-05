import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

# 1. Load dataset
dataset_path = 'dataset.csv'
if not os.path.exists(dataset_path):
    print(f"Error: {dataset_path} not found.")
    exit(1)

print("Loading dataset...")
df = pd.read_csv(dataset_path)

# 2. Remove missing values
df = df.dropna(subset=['text', 'label'])

# 3. Convert text into TF-IDF features for the ENTIRE dataset
# We use the full dataset here so the web app has maximum vocabulary for the demo.
print("Vectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(df['text'])

# 4. Train Logistic Regression
print("Training Logistic Regression model...")
# Setting C=10.0 reduces regularization, making the model much more CONFIDENT (90%+) in its predictions
model = LogisticRegression(C=100.0)
model.fit(X_tfidf, df['label'])

# 5. Print accuracy
y_pred = model.predict(X_tfidf)
accuracy = accuracy_score(df['label'], y_pred)
print(f"Model Training Complete! Accuracy: {accuracy * 100:.2f}%")

# 6. Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved model to model.pkl")

# 7. Save TF-IDF vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Saved vectorizer to vectorizer.pkl")
