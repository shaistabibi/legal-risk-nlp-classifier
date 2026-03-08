#  Legal Risk NLP Classifier

An end-to-end **Machine Learning + NLP system** that analyzes legal investment clauses and automatically detects **potentially risky language** using Natural Language Processing.

This project demonstrates how to design and deploy a **production-style ML pipeline** including data preprocessing, feature engineering, model training, evaluation, API serving, and a user interface.

---

# Project Overview

Legal contracts often contain complex investment clauses that may introduce **regulatory risks or compliance issues**.
Manually reviewing thousands of clauses is time-consuming and expensive.

This project builds an **AI-powered legal clause classifier** that predicts whether a clause is:

* **Safe / Compliant**
*  **Risky / Potential Issue**

The model is trained using a dataset of **investment-related legal clauses** and deployed through both:

* **FastAPI REST API**
* **Streamlit Web Interface**

---

# Machine Learning Pipeline

```
Legal Clauses Dataset
        ↓
Data Cleaning
        ↓
TF-IDF Feature Engineering
        ↓
Train/Test Split
        ↓
Logistic Regression Model
        ↓
Model Evaluation
        ↓
FastAPI Inference API
        ↓
Streamlit User Interface
```

---

#  Model Performance

Evaluation on the test dataset:

| Metric    | Score    |
| --------- | -------- |
| Accuracy  | **98%**  |
| Precision | **0.99** |
| Recall    | **0.97** |
| F1 Score  | **0.98** |

Confusion Matrix:

```
[[1632   16]
 [  68 2513]]
```

The model successfully detects **97% of risky clauses**, making it suitable for **legal AI assistance tools**.

---

# Tech Stack

**Languages & Libraries**

* Python
* Pandas
* Scikit-learn
* TF-IDF NLP
* Logistic Regression

**AI Infrastructure**

* FastAPI (Model Serving API)
* Streamlit (Interactive Web Interface)
* Pickle (Model Serialization)

**Development Tools**

* Git & GitHub
* Virtual Environment
* Modular ML Project Architecture

---

#  Project Structure

```
legal-risk-nlp-classifier
│
├── app
│   └── streamlit_app.py        # Web interface
│
├── data
│   └── raw
│       └── legal_docs_modified.csv
│
├── models
│   ├── legal_model.pkl
│   └── vectorizer.pkl
│
├── src
│   ├── data
│   │   └── preprocess.py
│   │
│   ├── features
│   │   └── vectorizer.py
│   │
│   ├── models
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   └── api
│       └── app.py
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/legal-risk-nlp-classifier.git
cd legal-risk-nlp-classifier
```

Create virtual environment:

```
python -m venv .venv
```

Activate environment:

Windows:

```
.venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Run the ML Pipeline

Train the model:

```
python -m src.models.train
```

Evaluate model:

```
python -m src.models.evaluate
```

---

# Run the FastAPI Service

Start the API server:

```
uvicorn src.api.app:app --reload
```

Open API documentation:

```
http://127.0.0.1:8000/docs
```

Example request:

```
POST /predict
```

Input:

```json
{
  "clause": "The company guarantees fixed investment returns regardless of market risk."
}
```

Output:

```json
{
  "prediction": "Risky Clause"
}
```

---

# Run the Web Interface

Start the Streamlit application:

```
streamlit run app/streamlit_app.py
```

The web interface allows users to **paste legal clauses and receive AI predictions instantly**.

---

# Example Use Cases

* Legal contract risk screening
* Automated compliance monitoring
* Investment agreement analysis
* Legal document summarization pipelines
* AI assistants for legal professionals

---

# Future Improvements

Planned upgrades to improve this project:

* Replace TF-IDF with **BERT / Transformer-based legal NLP**
* Add **MLflow experiment tracking**
* Implement **Docker containerization**
* Deploy API using **cloud services**
* Integrate **Retrieval-Augmented Generation (RAG)** for legal document analysis

---

#  Author
Shaista
AI / Machine Learning Engineer Portfolio Project.

Focused on building **real-world AI systems including NLP, RAG pipelines, and production ML APIs**.

---

# ⭐ If you find this project useful

Please consider **starring the repository** on GitHub.

