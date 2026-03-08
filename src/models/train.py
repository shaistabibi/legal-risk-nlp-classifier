import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.features.vectorizer import create_vectorizer


# Load dataset
df = pd.read_csv("legal_docs_modified.csv")
print(df.head())
print(df.columns)


# Remove rows where clause_text is empty
df = df.dropna(subset=["clause_text"])

# Convert text to string
df["clause_text"] = df["clause_text"].astype(str)
X = df["clause_text"]
y = df["clause_status"]

#TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#VECTORIZATION
vectorizer = create_vectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train_vec, y_train)

#SAVE MODEL 

pickle.dump(model, open("models/legal_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("✅ Model training completed")