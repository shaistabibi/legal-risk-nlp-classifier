import pandas as pd
import pickle

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("legal_docs_modified.csv")

# Clean data
df = df.dropna(subset=["clause_text"])
df["clause_text"] = df["clause_text"].astype(str)
X = df["clause_text"]
y = df["clause_status"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load saved vectorizer
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Convert text to vectors
X_test_vec = vectorizer.transform(X_test)

# Load trained model
model = pickle.load(open("models/legal_model.pkl", "rb"))
predictions = model.predict(X_test_vec)
# Confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, predictions))