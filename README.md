# Week 2 Deliverable: Semantic Search

A semantic search implementation for the AWS Student Builder Group Week 2 project.

## Features

- Loading FAQ data from CSV
- Text cleaning (lowercase, punctuation removal)
- Overlapping text chunking
- Semantic embeddings using sentence-transformers
- Vector search using ChromaDB
- Pretty-printed search results

## Setup

### Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the semantic search script:

```bash
python semantic_search.py
```

This will:

- Load FAQ data from `faq.csv`
- Clean and chunk the text
- Generate embeddings using `all-MiniLM-L6-v2`
- Store chunks in ChromaDB
- Run 3 example searches and display results

## Functions

- `data_load(path)` — Load FAQ data from CSV
- `clean_text(text)` — Clean text (lowercase, remove punctuation)
- `chunk_text(text, chunk_size, overlap)` — Split text into overlapping chunks
- `get_embeddings(texts)` — Generate embeddings using sentence-transformers
- `setup_vector_db()` — Set up and populate ChromaDB with FAQ chunks
- `search_similar(query, n_results)` — Search for similar documents
- `print_search_results(query, results)` — Pretty-print search results

## Project Structure

```text
week2-deliverable/
├── data/
│   └── faq.csv              # FAQ dataset (20 questions)
├── semantic_search.py       # Main script with all functions
├── requirements.txt         # Dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Example Output

```text
Query: How do I install python?
============================================================
Result 1 (distance: 0.2341)
Question: What is Python and why is it popular for AI?
Answer: Python is a high-level programming language known for its simplicity...
------------------------------------------------------------

Result 2 (distance: 0.3122)
Question: How do virtual environments help in Python projects?
Answer: Virtual environments create isolated spaces for Python projects...
------------------------------------------------------------
```

## Dependencies

- pandas
- sentence-transformers
- chromadb

## Repository

GitHub: https://github.com/Sarah-Tuemay/week2-semantic-search
