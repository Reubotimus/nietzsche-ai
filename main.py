# nietzsche_rag.py
"""Retrieval‑Augmented Generation (RAG) system for chatting about
Friedrich Nietzsche, using his primary works and secondary essays
stored **as PDFs** in a Neon (serverless Postgres) + pgvector database.

Dependencies (install via pip):
    pip install langchain langchain-community pgvector psycopg2-binary pdfminer.six \
                tiktoken openai python-dotenv

Before first run, set environment variables (e.g. in a .env file):
    OPENAI_API_KEY=<your OpenAI key>
    NEON_DB_URL=postgresql://<user>:<password>@<host>/<db>?sslmode=require

Create the pgvector extension once per database:
    CREATE EXTENSION IF NOT EXISTS vector;

Directory layout (relative to project root):
    sources/
        primary/   # Nietzsche's own works (PDFs)
        secondary/ # Scholarly essays / commentaries (PDFs)

Run the index build + chat CLI:
    python nietzsche_rag.py  # first run will build the vector store

Use CLI commands inside the chat:
    reset   – clear conversation memory
    quit    – exit program
"""
from __future__ import annotations

import os
import pathlib
import logging
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

PRIMARY_DIR = pathlib.Path("./sources/primary")
SECONDARY_DIR = pathlib.Path("./sources/secondary")
EMBEDDING_DIM = 1536  # OpenAI text‑embedding‑3‑small
COLLECTION_NAME = "nietzsche_docs"
CHUNK_SIZE = 1_000      # characters
CHUNK_OVERLAP = 200     # characters
K_RETRIEVE = 6          # passages per query
MODEL_NAME = "gpt-4o"   # Any chat model your key supports
TEMPERATURE = 0.1
MAX_TOKENS = 512

load_dotenv()
DB_CONN_STR = os.getenv("NEON_DB_URL")
if not DB_CONN_STR:
    raise EnvironmentError("NEON_DB_URL not set; see README header for details.")

# ---------------------------------------------------------------------------
# Helpers – PDF loading & chunking
# ---------------------------------------------------------------------------

def _load_pdf_dir(directory: pathlib.Path, tier: str) -> List[Document]:
    """Load every PDF in *directory*, tagging metadata['tier']=tier."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    docs: List[Document] = []
    pdf_files = list(directory.rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"Loading PDF {i}/{len(pdf_files)}: {pdf_path.name}")
        for page_doc in PyPDFLoader(str(pdf_path)).load():
            page_doc.metadata["tier"] = tier
            page_doc.metadata["source_path"] = str(pdf_path)
            docs.append(page_doc)
        logger.info(f"Loaded {len(docs)} pages from {pdf_path.name}")
    return docs


def _split_documents(documents: List[Document]):
    logger.info(f"Starting document chunking for {len(documents)} pages")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks

# ---------------------------------------------------------------------------
# Build or connect to pgvector store
# ---------------------------------------------------------------------------

def build_or_load_pgvector(force_rebuild: bool = False) -> PGVector:
    """Return a PGVector vector store, creating + populating if empty/forced."""

    logger.info("Initializing OpenAI embeddings")
    embeddings = OpenAIEmbeddings()
    store = PGVector(
        embeddings=embeddings,
        connection=DB_CONN_STR,
        collection_name=COLLECTION_NAME,
        distance_strategy="cosine",
        use_jsonb=True,
        pre_delete_collection=force_rebuild,
    )

    # If collection is empty OR user requests rebuild, (re)insert all docs
    engine = create_engine(DB_CONN_STR, future=True)
    with Session(engine) as sess:
        has_collection = store.get_collection(sess) is not None

    if force_rebuild or not has_collection:
        logger.info("Building/refreshing pgvector collection")
        logger.info("Loading primary source documents...")
        primary_docs = _load_pdf_dir(PRIMARY_DIR, "primary")
        logger.info("Loading secondary source documents...")
#        secondary_docs = _load_pdf_dir(SECONDARY_DIR, "secondary")
        docs = _split_documents(primary_docs)
        if has_collection:
            logger.info("Deleting existing collection")
            store.delete_collection()
        logger.info(f"Adding {len(docs)} document chunks to vector store")
        store.add_documents(docs)  # ids auto-generated
        logger.info("Vector store population complete")

    return store

# ---------------------------------------------------------------------------
# Service layer
# ---------------------------------------------------------------------------

class NietzscheRAGService:
    """High‑level façade combining retrieval and generation for chat."""

    def __init__(
        self,
        vector_store: Optional[PGVector] = None,
        k: int = K_RETRIEVE,
        model_name: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        logger.info("Initializing NietzscheRAGService")
        if vector_store is None:
            logger.info("No vector store provided, building new one")
            vector_store = build_or_load_pgvector()

        logger.info(f"Setting up retriever with k={k}")
        self.retriever = vector_store.as_retriever(search_kwargs={"k": k})
        logger.info(f"Initializing LLM with model={model_name}")
        self.llm = ChatOpenAI(model=model_name,
                              temperature=temperature,
                              max_tokens=max_tokens)
        self.memory = ConversationBufferMemory(memory_key="chat_history",
                                               return_messages=True)
        logger.info("Setting up QA chain")
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            verbose=False,
        )
        logger.info("NietzscheRAGService initialization complete")

    # ----------------‑ Public API

    def ask(self, query: str) -> str:
        """Answer *query*, returning the model's answer."""
        logger.info(f"Processing query: {query[:50]}...")
        response = self.qa_chain({"question": query})
        logger.info("Query processed successfully")
        return response["answer"].strip()

    def reset_memory(self):
        """Clear conversation memory (start fresh dialogue)."""
        logger.info("Clearing conversation memory")
        self.memory.clear()

# ---------------------------------------------------------------------------
# CLI / demo interface
# ---------------------------------------------------------------------------

def run_cli():
    print("≡ Nietzsche RAG Chat (pgvector) ≡")
    print("Type 'quit' to exit, 'reset' to clear memory.\n")

    svc = NietzscheRAGService()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        lower = user_input.lower()
        if lower in {"quit", "exit"}:
            print("Goodbye!")
            break
        if lower == "reset":
            svc.reset_memory()
            print("Memory cleared.\n")
            continue

        answer = svc.ask(user_input)
        print(f"Bot: {answer}\n")

def analyze_chunking():
    """Analyze the chunking process and print statistics without running embeddings."""
    print("\n=== Document Analysis ===")
    
    # Load documents
    print("\nLoading documents...")
    primary_docs = _load_pdf_dir(PRIMARY_DIR, "primary")
    secondary_docs = _load_pdf_dir(SECONDARY_DIR, "secondary")
    all_docs = primary_docs + secondary_docs
    
    print(f"\nTotal documents loaded:")
    print(f"Primary sources: {len(primary_docs)} pages")
    print(f"Secondary sources: {len(secondary_docs)} pages")
    print(f"Total pages: {len(all_docs)}")
    
    # Calculate total text size
    total_chars = sum(len(doc.page_content) for doc in all_docs)
    total_mb = total_chars / (1024 * 1024)
    print(f"\nTotal text size: {total_chars:,} characters ({total_mb:.2f} MB)")
    
    # Split into chunks
    print("\nSplitting into chunks...")
    chunks = _split_documents(all_docs)
    
    # Analyze chunks
    avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
    print(f"\nChunking statistics:")
    print(f"Total chunks: {len(chunks):,}")
    print(f"Average chunk size: {avg_chunk_size:.0f} characters")
    print(f"Estimated tokens (1.3 tokens/char): {int(total_chars * 1.3):,}")
    
    # Estimate embedding costs
    estimated_tokens = total_chars * 1.3
    embedding_cost = (estimated_tokens / 1000) * 0.00002
    print(f"\nEstimated embedding costs:")
    print(f"Cost at $0.00002/1K tokens: ${embedding_cost:.2f}")
    
    # Estimate storage
    vector_size = 1536 * 4  # dimensions * bytes per float32
    total_vector_storage = len(chunks) * vector_size
    total_storage_mb = (total_vector_storage + total_chars) / (1024 * 1024)
    print(f"\nEstimated storage requirements:")
    print(f"Vector storage: {total_vector_storage / (1024 * 1024):.2f} MB")
    print(f"Total storage (with text): {total_storage_mb:.2f} MB")

if __name__ == "__main__":
    # Comment out the original main execution
    force = os.getenv("REBUILD_INDEX", "false").lower() == "true"
    build_or_load_pgvector(force_rebuild=force)  # ensure collection exists
    run_cli()
    
