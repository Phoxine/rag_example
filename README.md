# LangChain RAG Example

This project demonstrates how to build Retrieval Augmented Generation (RAG) applications using LangChain based on the [official LangChain RAG tutorial](https://docs.langchain.com/oss/python/langchain/rag).

## Overview

RAG applications consist of two main components:

1. **Indexing**:
   - Load data from sources
   - Split documents into manageable chunks
   - Embed chunks and store them in a vector store

2. **Retrieval & Generation**:
   - Retrieve relevant documents based on user queries
   - Pass retrieved context to a language model
   - Generate answers

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Install Dependencies

```bash
pip install -r requirements.txt
```

The requirements include:
- `langchain` ecosystem for RAG implementation
- `chromadb` for vector storage
- `sentence-transformers` and `langchain-huggingface` for Hugging Face embeddings support
- `openai` for OpenAI API access

### Set API Key

Configure your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or the scripts will prompt you to enter it when run.

## Embedding Configuration

This project supports two embedding providers:

- **OpenAI Embeddings** (default): Uses `text-embedding-3-small` model. Requires OpenAI API key.
- **Hugging Face Embeddings**: Uses local `sentence-transformers/all-MiniLM-L6-v2` model. No API key required.

To switch between embeddings, modify the `EMBEDDING_TYPE` variable in each script:
- Set `EMBEDDING_TYPE = "openai"` for OpenAI embeddings
- Set `EMBEDDING_TYPE = "huggingface"` for Hugging Face embeddings

**Note**: Hugging Face embeddings run locally and may be slower on first use while downloading the model.

## Project Structure

### 1. `1_indexing.py` - Indexing Pipeline

Demonstrates the indexing portion of a RAG application:

- **Load Documents**: Uses `WebBaseLoader` to load content from two web URLs (agent and prompt engineering posts)
- **Split Documents**: Uses `RecursiveCharacterTextSplitter` to split large documents into 500-character chunks with 100-character overlap
- **Store Documents**: Embeds chunks using configurable embeddings (OpenAI or Hugging Face) and stores them in `Chroma` when available; otherwise the code falls back to `InMemoryVectorStore`

**Run with**:
```bash
python 1_indexing.py
```

### 2. `2_rag_agent.py` - RAG Agent

Demonstrates an intelligent RAG agent with tool-based retrieval:

- **Smart Vector Store Loading**: Loads existing `Chroma` database if available; otherwise creates a new one using configurable embeddings (OpenAI or Hugging Face) from two blog posts with optimized 500-char chunks
- **Retrieval Tool**: Uses LangChain's `create_tool_calling_agent` with a custom retrieval tool
- **Intelligent Decisions**: Agent decides when and how to use the retrieval tool
- **Multi-step Reasoning**: Can execute multiple retrievals to answer complex questions

**Features**:
- ✅ Database persistence (reuses indexed data on subsequent runs)
- ✅ Smart search decisions (searches only when needed)
- ✅ Tool-based retrieval (LangChain agent framework)
- ✅ Flexible reasoning (multi-step problem solving)
- ⚡ Efficient (no re-indexing on repeat runs)

**Workflow**:
1. Check if `./chroma_db` exists → Load it (fast!)
2. Otherwise, index documents from web and save to `./chroma_db`
3. Create agent with retrieval tool
4. Run sample queries or enter interactive mode

**Run with**:
```bash
python 2_rag_agent.py
```

**First run** (creates database):
```bash
Building vector store...
Creating new vector store...
Loading documents from web...
Created and persisted Chroma vector store with 66 documents
```

**Subsequent runs** (loads existing database):
```bash
Building vector store...
Loading existing Chroma database from ./chroma_db...
Loaded existing Chroma database
```

### 3. `3_rag_chain.py` - RAG Chain

Demonstrates a simplified two-step RAG approach:

- **Smart Vector Store Loading**: Loads existing `Chroma` database if available; otherwise creates a new one using configurable embeddings (OpenAI or Hugging Face) from two blog posts with optimized 500-char chunks
- **Single-step Retrieval**: Always retrieves documents matching the user query
- **Single LLM Call**: Generates answer in one inference call
- **Low Latency**: Optimized for fast response times

**Workflow**:
1. Check if `./chroma_db` exists → Load it (fast!)
2. Otherwise, index documents from web and save to `./chroma_db`
3. For each query:
   - Retrieve relevant documents via semantic search
   - Pass documents as context to the LLM
   - Return the generated answer

**Features**:
- ✅ Database persistence (reuses indexed data on subsequent runs)
- ✅ Single inference call (low latency, cost-effective)
- ✅ Simple and reliable implementation
- ✅ Good for straightforward Q&A
- ❌ No intelligent search decisions (always searches)

**Run with**:
```bash
python 3_rag_chain.py
```

### Shared Database

Both `2_rag_agent.py` and `3_rag_chain.py` use the same `./chroma_db` directory:
- Run `1_indexing.py` first to create the database, OR
- Run either `2_rag_agent.py` or `3_rag_chain.py` (they'll create it if missing)
- Subsequent runs are **much faster** since they reuse the existing database

## Usage Examples

### Running the Indexing Pipeline

```bash
$ python 1_indexing.py
============================================================
RAG Indexing Pipeline
============================================================

Loading documents from web...
Loaded 2 document(s)
Total characters: 72333

Splitting documents...
Split into 225 sub-documents

Creating embeddings and storing documents...
Stored 225 documents in Chroma vector store
Chroma database persisted to ./chroma_db

Testing retrieval with a sample query...

Query: What is task decomposition?

Retrieved 2 documents:
...
```

### Running the RAG Agent (First Time)

```bash
$ python 2_rag_agent.py
Building vector store...
Creating new vector store...

Loading documents from web...
Splitting documents...
Created and persisted Chroma vector store with 225 documents

RAG Agent created successfully!
Running sample queries...
```

### Running the RAG Agent (Subsequent Times - Much Faster!)

```bash
$ python 2_rag_agent.py
Building vector store...
Loading existing Chroma database from ./chroma_db...
Loaded existing Chroma database

RAG Agent created successfully!
Running sample queries...
```

```
Agent thinking...

================================ Human Message =================================
What is task decomposition?
================================== Ai Message ==================================
Task decomposition is a technique used in AI systems...

```

### Running the RAG Chain

```bash
$ python 3_rag_chain.py
...
Your question: What is task decomposition?

Processing...

Answer: Task decomposition is the process of breaking down complex tasks into smaller, 
manageable subtasks...

Retrieved 4 relevant documents
```

## Concepts and Terminology

- **Document**: An object containing text content and metadata (source URL, document ID, chunk ID, content hash)
- **Embedding**: A vector representation of text content (supports OpenAI and Hugging Face models)
- **Vector Store**: A system for storing and querying embeddings
- **Retriever**: An object that finds relevant documents based on queries
- **Tool**: A function that agents can use
- **Agent**: An AI system that makes decisions based on tools and instructions

## Documentation

- [LangChain RAG Tutorial](https://docs.langchain.com/oss/python/langchain/rag)
- [LangChain Docs](https://docs.langchain.com/)
- [LangChain API Reference](https://reference.langchain.com/)

## Security Considerations

### Indirect Prompt Injection

RAG applications are susceptible to indirect prompt injection. Retrieved documents may contain text that resembles instructions.

**Mitigation Strategies**:
1. **Defensive Prompts**: Explicitly instruct the model to treat retrieved context as data only and ignore any instructions within it
2. **Wrap Context with Delimiters**: Use clear structural markers (e.g., XML tags like `<context>...</context>`) to separate data from instructions
3. **Validate Responses**: Check that the model's output matches expected format and handle unexpected formats gracefully

## Next Steps

- Experiment with different embedding models (OpenAI `text-embedding-3-large`, other Hugging Face models)
- Add [conversation memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) to support multi-turn interactions
- Add [long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory) across conversation threads
- Implement [structured responses](https://docs.langchain.com/oss/python/langchain/structured-output)
- Use [different vector stores](https://docs.langchain.com/oss/python/integrations/vectorstores) (Pinecone, Chroma, etc.)
- Build complex workflows with [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)

## License

MIT

## More Resources

- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangChain Academy](https://academy.langchain.com/)
- [LangSmith](https://smith.langchain.com/) - Monitor and debug LLM applications
