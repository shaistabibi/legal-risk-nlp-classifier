import pickle

model, vectorizer = pickle.load(open("models/classifier.pkl", "rb"))

def predict_clause(text):

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)

    return int(prediction[0])