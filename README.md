# Nietzsche AI - RAG System

A Retrieval-Augmented Generation (RAG) system specialized in Friedrich Nietzsche's works, combining primary sources with secondary literature to provide accurate and contextual responses about Nietzsche's philosophy.

## Overview

This project implements a sophisticated RAG system that:
- Processes both primary Nietzsche texts and secondary literature
- Uses vector embeddings for semantic search
- Implements a two-stage retrieval process for comprehensive context
- Provides answers with proper citations and source attribution

## Prerequisites

- Python 3.10+
- PostgreSQL database (using Neon.tech in this implementation)
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd nietzche-ai
```

2. Install the required dependencies:
```bash
pip install langchain-text-splitters langchain-community langgraph
pip install "langchain[openai]"
pip install langchain-core
pip install langchain-postgres
```

## Project Structure

```
nietzche-ai/
├── sources/
│   ├── primary/     # Primary Nietzsche texts (PDFs)
│   └── secondary/   # Secondary literature (PDFs)
├── generate-rag.ipynb
└── README.md
```

## Setup

1. Set up your OpenAI API key:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

2. Configure your PostgreSQL database connection (currently using Neon.tech)

## Usage

The system is implemented in a Jupyter notebook (`generate-rag.ipynb`) that demonstrates:
1. Loading and processing PDF documents
2. Creating vector embeddings
3. Implementing the RAG pipeline
4. Querying the system

To use the system:
1. Place your PDF documents in the appropriate directories (`sources/primary` and `sources/secondary`)
2. Run the notebook cells in sequence
3. Query the system using the provided interface

## Features

- **Dual-Source Retrieval**: Searches both primary and secondary sources
- **Contextual Understanding**: Uses GPT-4 for nuanced interpretation
- **Source Attribution**: Properly cites and references sources
- **Vector Storage**: Efficient semantic search using PostgreSQL vector store

## Dependencies

- langchain
- langchain-text-splitters
- langchain-community
- langgraph
- langchain-core
- langchain-postgres
- OpenAI API

## Notes

- The system uses `text-embedding-3-small` for embeddings
- Documents are split into chunks of 1000 characters with 200 character overlap
- The retrieval process fetches 3 relevant chunks from each source type