import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
def data_load(path):
    df = pd.read_csv(path)
    return df


def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text


def chunk_text(text, chunk_size=35, overlap=8):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk size")
    words = text.split()
    step = chunk_size - overlap

    return [" ".join(words[start: start + chunk_size]) for start in range(0, len(words), step)]


def get_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def search_similar(query, n_results=3):
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all_MiniLM-L6-v2")
    client = chromadb.chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="faq_collection",
        embedding_function=embedding_func
    )

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    return results


if __name__ == "__main__":
    df = data_load("data/faq.csv")
    df["clean_question"] = df["question"].apply(clean_text)
    df["clean_answer"] = df["answer"].apply(clean_text)
    print(df.head())
