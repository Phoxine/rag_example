"""
RAG Indexing Pipeline

This script demonstrates the indexing portion of a RAG application:
1. Load: Load documents from a web source
2. Split: Split documents into manageable chunks
3. Store: Embed and store chunks in a vector store
"""

import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    from langchain_core.vectorstores import InMemoryVectorStore
    CHROMA_AVAILABLE = False

# Embedding configuration
EMBEDDING_TYPE = "huggingface"  # Options: "openai" or "huggingface"

# Setup API key (only needed for OpenAI)
if EMBEDDING_TYPE == "openai" and not os.environ.get("OPENAI_API_KEY"):
    import getpass
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


def load_documents():
    """
    Load documents from web URLs using WebBaseLoader.
    We only keep post title, headers, and content from the HTML.
    """
    print("Loading documents from web...")
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
    ]
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    
    print(f"Loaded {len(docs)} document(s)")
    total_chars = sum(len(doc.page_content) for doc in docs)
    print(f"Total characters: {total_chars}")
    if docs:
        print(f"\nFirst 500 characters:\n{docs[0].page_content[:500]}\n")
    
    return docs


def split_documents(docs):
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    This is recommended for generic text use cases.
    """
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Smaller chunks for better precision
        chunk_overlap=100,
        add_start_index=True, # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)
    
    # Add metadata and deduplicate (very important!)
    from hashlib import md5
    seen_hashes = set()
    unique_docs = []

    for i, doc in enumerate(all_splits):
        content_hash = md5(doc.page_content.encode()).hexdigest()

        # Skip duplicates
        if content_hash in seen_hashes:
            continue

        source = doc.metadata.get("source", "unknown")

        doc.metadata.update({
            "source": source,
            "doc_id": md5(source.encode()).hexdigest(),
            "chunk_id": len(unique_docs),  # Use unique_docs length as chunk_id
            "doc_hash": content_hash,
        })

        seen_hashes.add(content_hash)
        unique_docs.append(doc)
    
    print(f"Split into {len(all_splits)} sub-documents")
    print(f"Deduplicated: {len(all_splits)} → {len(unique_docs)} unique chunks")
    print(f"\nExample chunk:\n{unique_docs[0].page_content[:200]}\n")
    
    return unique_docs


def store_documents(unique_docs):
    """
    Create embeddings and store documents in a vector store.
    """
    print("Creating embeddings and storing documents...")
    
    if EMBEDDING_TYPE == "openai":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    elif EMBEDDING_TYPE == "huggingface":
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unsupported EMBEDDING_TYPE: {EMBEDDING_TYPE}. Choose 'openai' or 'huggingface'.")
    
    if CHROMA_AVAILABLE:
        vector_store = Chroma.from_documents(
            documents=unique_docs,
            embedding=embeddings,
            persist_directory="./chroma_db",
        )
        vector_store.persist()
        print(f"Created Chroma DB with {len(unique_docs)} chunks")
        print("Persisted to ./chroma_db\n")
    else:
        vector_store = InMemoryVectorStore(embeddings)
        document_ids = vector_store.add_documents(documents=unique_docs)
        print(f"Created in-memory store with {len(unique_docs)} chunks\n")
    
    return vector_store


def test_retrieval(vector_store):
    """
    Test the vector store with a sample query.
    """
    print("Testing retrieval with sample queries...")
    query = "What is task decomposition?"
    # k = 2 means search for top 2 most similar chunks from the vector store
    retrieved_docs = vector_store.similarity_search(query, k=2)
    
    print(f"\nQuery: {query}")
    print(f"\nRetrieved {len(retrieved_docs)} documents:\n")
    for i, doc in enumerate(retrieved_docs):
        print(f"--- Document {i+1} ---")
        print(f"Content: {doc.page_content[:300]}...\n")

    print("==================================\n")


    query2 = "What is Chain-of-Thought"
    retrieved_docs = vector_store.similarity_search(query2, k=2)

    print(f"\nQuery: {query2}")
    print(f"\nRetrieved {len(retrieved_docs)} documents:\n")
    for i, doc in enumerate(retrieved_docs):
        print(f"--- Document {i+1} ---")
        print(f"Content: {doc.page_content[:300]}...\n")


def main():
    """Main indexing pipeline."""
    print("=" * 60)
    print("RAG Indexing Pipeline")
    print("=" * 60 + "\n")
    
    vector_store = None
    if CHROMA_AVAILABLE and os.path.exists("./chroma_db"):
        print("Loading existing Chroma database...")
        
        if EMBEDDING_TYPE == "openai":
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        elif EMBEDDING_TYPE == "huggingface":
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        else:
            raise ValueError(f"Unsupported EMBEDDING_TYPE: {EMBEDDING_TYPE}")
        
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./chroma_db",
        )
        print("Loaded existing DB (skip processing)\n")
    else:
        # Step 1: Load documents
        docs = load_documents()
        
        # Step 2: Split documents and deduplicate
        unique_docs = split_documents(docs)
        
        # Step 3: Store documents
        vector_store = store_documents(unique_docs)
    
    
    # Step 4: Test retrieval
    test_retrieval(vector_store)
    
    print("\n" + "=" * 60)
    print("Indexing complete! Vector store is ready for retrieval.")
    print("=" * 60)
    
    return vector_store


if __name__ == "__main__":
    vector_store = main()
