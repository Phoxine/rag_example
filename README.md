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
- **Split Documents**: Uses `RecursiveCharacterTextSplitter` to split large documents into 500-character chunks with 100-character overlap, and removes duplicates based on content hash
- **Store Documents**: Embeds chunks using configurable embeddings (OpenAI or Hugging Face) and stores them in `Chroma` when available; otherwise the code falls back to `InMemoryVectorStore`

**Run with**:
```bash
python 1_indexing.py
```

### 2. `2_rag_agent.py` - RAG Agent

Demonstrates an intelligent RAG agent with tool-based retrieval:

- **Smart Vector Store Loading**: Loads existing `Chroma` database if available; otherwise creates a new one using configurable embeddings (OpenAI or Hugging Face) from two blog posts with optimized 500-char chunks and deduplication
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

- **Smart Vector Store Loading**: Loads existing `Chroma` database if available; otherwise creates a new one using configurable embeddings (OpenAI or Hugging Face) from two blog posts with optimized 500-char chunks and deduplication
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

### 4. `4_incremental_index.py` - Incremental Indexing

Demonstrates incremental indexing to avoid re-processing existing documents:

- **Load Vector Store**: Loads existing Chroma database or creates new one if it doesn't exist
- **Check Existing Hashes**: Retrieves document hashes from metadata to identify already indexed content
- **Load New Documents**: Loads documents from provided URLs
- **Incremental Processing**: Only processes and adds new documents that haven't been indexed before
- **Test Retrieval**: Tests the updated vector store with sample queries

**Features**:
- ✅ Avoids duplicate processing (checks document hashes)
- ✅ Efficient updates (only indexes new content)
- ✅ Metadata tracking (adds indexing timestamps)
- ✅ Database persistence (updates existing Chroma database)
- ⚡ Fast for large existing databases

**Workflow**:
1. Load existing vector store from `./chroma_db`
2. Extract existing document hashes from metadata
3. Load new documents from URLs
4. Filter out already indexed documents
5. Split and embed only new documents
6. Add new chunks to vector store
7. Test retrieval with sample queries

**Run with**:
```bash
python 4_incremental_index.py
```

### Shared Database

All scripts (`1_indexing.py`, `2_rag_agent.py`, `3_rag_chain.py`, and `4_incremental_index.py`) use the same `./chroma_db` directory:
- Run `1_indexing.py` first to create the initial database, OR
- Run any of the other scripts (they'll create it if missing)
- `4_incremental_index.py` can be used to add new documents without re-processing existing ones
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

### Running the Incremental Indexing

```bash
$ python 4_incremental_index.py
============================================================
Incremental RAG Indexing Pipeline
============================================================

Loading documents from web...
Loaded 2 document(s)
Total characters: 72333

First 500 characters:
...

Splitting documents...
Split into 225 sub-documents
Deduplicated: 225 → 225 unique chunks

Loaded existing vector store from ./chroma_db
No new documents to index

Testing retrieval with sample queries...

Query: What is task decomposition?

Retrieved 2 documents:
--- Document 1 ---
Content: Task decomposition is a technique used in AI systems...
...
==================================

Query: What is Chain-of-Thought
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

below is the real output from running `3_rag_chain.py`:

```bash
$ python 3_rag_chain.py
...
RAG Chain created successfully!
Running sample queries...

============================================================
RAG Chain - Running Sample Queries
============================================================

--- Query 1 ---
Question: What is task decomposition?

Answer: Task decomposition is the process of breaking down complex tasks into smaller, manageable subgoals. This can be achieved through various methods, such as prompting a language model, using task-specific instructions, or incorporating human inputs. It enables more efficient handling of complicated tasks and enhances overall performance.

Retrieved 4 relevant documents
------------------------------------------------------------

--- Query 2 ---
Question: What are the key components of an autonomous agent?

Answer: In a LLM-powered autonomous agent system, the key components include the large language model (LLM) as the agent's brain, external APIs for additional information, and tools for tasks like code execution and accessing proprietary information. These components work together to enhance the agent's capabilities beyond just generating text.

Retrieved 4 relevant documents
------------------------------------------------------------

--- Query 3 ---
Question: How do agents use memory?

Answer: Agents use memory through short-term and long-term memory systems. Short-term memory involves in-context learning, while long-term memory allows agents to retain and recall information over extended periods, often utilizing an external vector store for fast retrieval. This combination enables agents to behave based on past experiences and interact with other agents effectively.

Retrieved 4 relevant documents
------------------------------------------------------------

--- Query 4 ---
Question: What is the difference between CoT and ReAct?

Answer: CoT (Chain-of-Thought) prompting generates a sequence of short sentences to describe reasoning step by step, while ReAct integrates reasoning and acting by combining task-specific actions with natural language reasoning. ReAct allows LLMs to interact with the environment through discrete actions, whereas CoT focuses on generating reasoning chains. Essentially, ReAct extends the action space beyond just reasoning, enabling more dynamic interactions.

Retrieved 4 relevant documents
------------------------------------------------------------

Would you like to run in interactive mode? (yes/no): no
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

- **Use incremental indexing**: Run `4_incremental_index.py` to add new documents without re-processing existing ones
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
