# Week 2 Deliverable: Semantic Search

A semantic search implementation for the AWS Student Builder Group Week 2 project.

## Features

- Loading FAQ data from CSV
- Text cleaning (lowercase, punctuation removal)
- Text chunking
- Placeholder for embeddings and vector search

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

## Project Structure

```
week2-deliverable/
├── data/
│   └── faq.csv          # FAQ dataset
├── semantic_search.py   # Main script with functions
├── requirements.txt     # Dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Functions

- `data_load(path)` - Load FAQ data from CSV
- `clean_text(text)` - Clean text (lowercase, remove punctuation)
- `chunk_text(text, max_length)` - Split text into chunks
- `get_embeddings(texts)` - Generate embeddings (placeholder)
- `search_similar(query, n_results)` - Search similar documents (placeholder)

## Next Steps

- Integrate sentence-transformers for embeddings
- Use ChromaDB for vector storage and search
- Complete the semantic search pipeline

## Repository

GitHub: [Semantic Search](https://github.com/Sarah-Tuemay/week2-semantic-search)
