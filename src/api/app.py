from fastapi import FastAPI
import pickle

app = FastAPI()

# Load model
model = pickle.load(open("models/legal_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


@app.get("/")
def home():
    return {"message": "Legal Risk NLP API Running"}


@app.post("/predict")
def predict_risk(clause: str):

    # Convert text to vector
    clause_vec = vectorizer.transform([clause])

    # Predict
    prediction = model.predict(clause_vec)[0]

    if prediction == 1:
        result = "Risky Clause"
    else:
        result = "Safe Clause"

    return {"prediction": result}