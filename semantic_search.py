import pandas as pd
import textwrap

def data_load(path):
    df = pd.read_csv(path)
    return df

def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def chunk_text(text, max_length=100):

    chunks = textwrap.wrap(text, width=max_length)
    return chunks

def get_embeddings(texts):
    pass

def search_similar(query, n_results=3):
    pass

if __name__ == "__main__":
    df = data_load("data/faq.csv")
    df["clean_question"] = df["question"].apply(clean_text)
    df["clean_answer"] = df["answer"].apply(clean_text)
    print(df.head())
