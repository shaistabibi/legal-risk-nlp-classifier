import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("models/legal_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

st.title("⚖️ Legal Risk NLP Classifier")

st.write("Enter a legal clause to detect if it is risky or safe.")

clause = st.text_area("Legal Clause")

if st.button("Analyze Clause"):

    if clause.strip() == "":
        st.warning("Please enter a clause.")
    else:
        clause_vec = vectorizer.transform([clause])
        prediction = model.predict(clause_vec)[0]

        if prediction == 1:
            st.error("⚠️ Risky Clause Detected")
        else:
            st.success("✅ Safe Clause")