import os
import hashlib
import datetime
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


def load_documents(urls):
    """
    Load documents from web URLs using WebBaseLoader.
    We only keep post title, headers, and content from the HTML.
    """
    print("Loading documents from web...")
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
    seen_hashes = set()
    unique_docs = []

    for i, doc in enumerate(all_splits):
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()

        # Skip duplicates
        if content_hash in seen_hashes:
            continue

        source = doc.metadata.get("source", "unknown")

        doc.metadata.update({
            "source": source,
            "doc_id": hashlib.md5(source.encode()).hexdigest(),
            "chunk_id": len(unique_docs),  # Use unique_docs length as chunk_id
            "doc_hash": content_hash,
        })

        seen_hashes.add(content_hash)
        unique_docs.append(doc)

    print(f"Split into {len(all_splits)} sub-documents")
    print(f"Deduplicated: {len(all_splits)} → {len(unique_docs)} unique chunks")
    print(f"\nExample chunk:\n{unique_docs[0].page_content[:200]}\n")

    return unique_docs


def load_vector_store(vector_store_path="./chroma_db", embedding_provider="openai"):
    """
    Load an existing vector store from the specified path.
    """
    if not CHROMA_AVAILABLE:
        raise ValueError("Chroma is not available. Install chromadb.")

    if embedding_provider == "openai":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        if not os.environ.get("OPENAI_API_KEY"):
            import getpass
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

    elif embedding_provider == "huggingface":
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unsupported embedding_provider: {embedding_provider}")

    if os.path.exists(vector_store_path):
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=vector_store_path,
        )
        print(f"Loaded existing vector store from {vector_store_path}")
    else:
        raise Exception("No existing vector store found.")

    return vector_store


def incremental_index(urls, vector_store_path="./chroma_db", embedding_provider="openai"):
    """Incrementally index new documents by checking existing document hashes."""
    # Load existing vector store
    vector_store = load_vector_store(vector_store_path, embedding_provider)

    # Get existing document hashes from metadata
    existing_hashes = set()
    if hasattr(vector_store, '_collection'):
        try:
            results = vector_store._collection.get(include=['metadatas'])
            for metadata in results['metadatas']:
                if 'doc_hash' in metadata:
                    existing_hashes.add(metadata['doc_hash'])
        except:
            pass  # No existing collection or error
    print(f"Number of existing document hashes: {len(existing_hashes)}")
    # Load and process new documents
    docs = load_documents(urls)
    new_docs = []

    for doc in docs:
        doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        if doc_hash not in existing_hashes:
            # Add metadata
            doc.metadata['doc_hash'] = doc_hash
            doc.metadata['indexed_at'] = datetime.datetime.now().isoformat()
            new_docs.append(doc)

    if new_docs:
        # Split and add new documents only
        split_docs = split_documents(new_docs)
        vector_store.add_documents(split_docs)
        print(f"Added {len(split_docs)} new document chunks")
    else:
        print("No new documents to index")

    return vector_store


def test_retrieval(vector_store):
    """
    Test the vector store with a sample query.
    """
    print("Testing retrieval with sample queries...")
    query = "What is Knowledge Distillation?"
    # k = 2 means search for top 2 most similar chunks from the vector store
    retrieved_docs = vector_store.similarity_search(query, k=2)

    print(f"\nQuery: {query}")
    print(f"\nRetrieved {len(retrieved_docs)} documents:\n")
    for i, doc in enumerate(retrieved_docs):
        print(f"--- Document {i+1} ---")
        print(f"Content: {doc.page_content[:300]}...\n")



def main():
    """Main incremental indexing pipeline."""
    print("=" * 60)
    print("Incremental RAG Indexing Pipeline")
    print("=" * 60 + "\n")

    # Example URLs to index incrementally
    urls = [
        "https://lilianweng.github.io/posts/2023-01-10-inference-optimization/"
    ]

    # Run incremental indexing
    vector_store = incremental_index(urls, vector_store_path="./chroma_db", embedding_provider="huggingface")

    # Test retrieval
    test_retrieval(vector_store)

    print("\n" + "=" * 60)
    print("Incremental indexing complete! Vector store is ready for retrieval.")
    print("=" * 60)

    return vector_store


if __name__ == "__main__":
    vector_store = main()