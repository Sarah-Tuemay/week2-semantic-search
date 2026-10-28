import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions


def data_load(path):
    """Load FAQ data from CSV file."""
    df = pd.read_csv(path)
    return df


def clean_text(text):
    """Clean text: lowercase, remove punctuation, strip whitespace."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text


def chunk_text(text, chunk_size=35, overlap=8):
    """Split text into overlapping chunks of words."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk size")
    
    words = text.split()
    step = chunk_size - overlap

    return [" ".join(words[start: start + chunk_size]) for start in range(0, len(words), step)]


def get_embeddings(texts):
    """Generate embeddings for a list of texts."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def setup_vector_db():
    """Set up ChromaDB and store all FAQ chunks."""
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="faq_collection",
        embedding_function=embedding_func
    )

    if collection.count() > 0:
        return collection
    

    df = data_load("data/faq.csv")
    df["clean_answer"] = df["answer"].apply(clean_text)
    df["chunks"] = df["answer"].apply(lambda x: chunk_text(x))
    

    all_chunks = []
    for idx, row in df.iterrows():
        for i, chunk in enumerate(row["chunks"]):
            all_chunks.append({
                "id": f"q{idx}_chunk{i}",
                "text": chunk,
                "question": row["question"],
                "answer": row["answer"]
            })
    
    # Add to ChromaDB
    collection.add(
        documents=[c["text"] for c in all_chunks],
        ids=[c["id"] for c in all_chunks],
        metadatas=[{"question": c["question"], "answer": c["answer"]} for c in all_chunks]
    )
    
    print(f"Stored {len(all_chunks)} chunks in ChromaDB")
    return collection


def search_similar(query, n_results=3):
    """Search for similar documents in vector database."""
    # Set up or get existing collection
    collection = setup_vector_db()
    
    # Search for similar documents
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    return results


def print_search_results(query, results, max_answer_length=200):
    """Pretty print search results."""
    print(f"\nQuery: {query}")
    print("=" * 60)
    
    if not results['documents'] or not results['documents'][0]:
        print("No results found.")
        return
    
    for i, (doc, distance, metadata) in enumerate(zip(
        results['documents'][0],
        results['distances'][0],
        results['metadatas'][0]
    ), 1):
        print(f"\nResult {i} (distance: {distance:.4f})")
        print(f"Question: {metadata['question']}")
        answer = metadata['answer']
        if len(answer) > max_answer_length:
            answer = answer[:max_answer_length] + "..."
        print(f"Answer: {answer}")
        print("-" * 60)


if __name__ == "__main__":
    # Load and prepare data
    print("Loading FAQ data...")
    df = data_load("data/faq.csv")
    df["clean_question"] = df["question"].apply(clean_text)
    df["clean_answer"] = df["answer"].apply(clean_text)
    df["chunks"] = df["answer"].apply(lambda x: chunk_text(x))
    
    print(f"Loaded {len(df)} FAQ entries")
    print(f"Total chunks: {sum(len(chunks) for chunks in df['chunks'])}")
    
    # Generate and display embeddings
    all_texts = list(df["clean_question"]) + list(df["clean_answer"])
    print("\nGenerating embeddings...")
    embeddings = get_embeddings(all_texts)
    print(f"Generated {len(embeddings)} embeddings (shape: {embeddings.shape})")
    
    # Search examples
    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH EXAMPLES")
    print("=" * 60)
    
    # Example 1
    results = search_similar("How do I install python?")
    print_search_results("How do I install python?", results)
    
    # Example 2
    results = search_similar("What is the difference between Python and JavaScript?")
    print_search_results("What is the difference between Python and JavaScript?", results)
    
    # Example 3
    results = search_similar("How does Git work for version control?")
    print_search_results("How does Git work for version control?", results)
    
    print("\n" + "=" * 60)
    print("Search complete!")
    print("=" * 60)