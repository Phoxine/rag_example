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

### Set API Key

Configure your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or the scripts will prompt you to enter it when run.

## Project Structure

### 1. `1_indexing.py` - Indexing Pipeline

Demonstrates the indexing portion of a RAG application:

- **Load Documents**: Uses `WebBaseLoader` to load content from web URLs
- **Split Documents**: Uses `RecursiveCharacterTextSplitter` to split large documents into 1000-character chunks with 200-character overlap
- **Store Documents**: Embeds chunks using OpenAI embeddings and stores them in `Chroma` when available; otherwise the code falls back to `InMemoryVectorStore`

**Run with**:
```bash
python 1_indexing.py
```

### 2. `2_rag_agent.py` - RAG Agent

Demonstrates retrieval and generation using an agent:

- **Retrieval Tool**: Defines a tool that the agent can use to retrieve context
- **Smart Search**: Agent decides when and what to search for
- **Multi-step Reasoning**: Agent can execute multiple searches to support a single user query

**Features**:
- ✅ Search only when needed (handles greetings and follow-ups without searching)
- ✅ Contextual search queries (LLM crafts queries incorporating context)
- ✅ Multiple searches allowed
- ❌ Two inference calls (one to generate query, one for answer)
- ❌ Reduced control (LLM may skip necessary searches)

**Run with**:
```bash
python 2_rag_agent.py
```

### 3. `3_rag_chain.py` - RAG Chain

Demonstrates a simplified two-step RAG chain:

- **Retrieve**: Always search using the user query
- **Generate**: Pass retrieved context with the query in a single LLM call
- **Single Inference**: One LLM call per query for lower latency

**Features**:
- ✅ Single inference call (low latency)
- ✅ Simple and straightforward implementation
- ✅ Good for simple, constrained queries
- ❌ Always searches (less flexible)
- ❌ Unconditional context (doesn't consider conversation state)

**Run with**:
```bash
python 3_rag_chain.py
```

## Usage Examples

### Running the Indexing Pipeline

```bash
$ python 1_indexing.py
============================================================
RAG Indexing Pipeline
============================================================

Loading documents from web...
Loaded 1 document(s)
Total characters: 43131

Splitting documents...
Split into 66 sub-documents

Creating embeddings and storing documents...
Stored 66 documents in vector store

Testing retrieval with a sample query...

Query: What is task decomposition?

Retrieved 2 documents:
...
```

### Running the RAG Agent

```bash
$ python 2_rag_agent.py
...
Your question: What is task decomposition?

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

- **Document**: An object containing text content and metadata
- **Embedding**: A vector representation of text content
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
