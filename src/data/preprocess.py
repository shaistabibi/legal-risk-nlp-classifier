import pandas as pd
import re
import string


df=pd.read_csv('legal_docs_modified.csv')
print(df)

def clean_text(text):

    text = text.lower()

    text = re.sub(r'\d+', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    return text