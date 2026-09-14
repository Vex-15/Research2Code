# Research2Code

A Retrieval-Augmented Generation (RAG) application for analyzing research papers and providing implementation-focused explanations.

Research2Code allows users to upload a research paper in PDF format and interact with it through natural language. The application retrieves relevant sections from the paper and uses a large language model to generate context-aware responses.

## Features

* Upload research papers in PDF format
* Extract and process document text
* Split documents into semantic chunks
* Generate embeddings locally using Hugging Face
* Store embeddings in ChromaDB
* Perform semantic similarity search
* Generate context-grounded responses using Groq
* Ask methodology, algorithm, results, and implementation-related questions
* View retrieved source context

## Architecture

```text
Research Paper
      |
      v
PDF Text Extraction
      |
      v
Document Chunking
      |
      v
Hugging Face Embeddings
      |
      v
ChromaDB Vector Store
      |
      v
Semantic Retrieval
      |
      v
Relevant Context
      |
      v
Groq LLM
      |
      v
Generated Response
```

## How It Works

### 1. Document Processing

The uploaded PDF is processed using PyPDF2 to extract its text content.

### 2. Text Chunking

The extracted text is divided into smaller overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.

### 3. Embedding Generation

Each text chunk is converted into a vector representation using the `all-MiniLM-L6-v2` embedding model from Hugging Face.

### 4. Vector Storage

The generated embeddings are stored in ChromaDB, which is used as the application's vector store.

### 5. Retrieval

When a user submits a question, the application performs semantic similarity search to retrieve the most relevant document chunks.

### 6. Response Generation

The retrieved context is provided to a Groq-hosted Llama model, which generates a response based on the relevant sections of the research paper.

## Example Queries

```text
What problem does this paper address?

Explain the proposed methodology.

What algorithm does the paper use?

What are the key experimental results?

What are the limitations of the proposed approach?

How can I implement the proposed method?

What libraries would be required to reproduce the approach?
```

## Technology Stack

| Technology   | Purpose                              |
| ------------ | ------------------------------------ |
| Python       | Application development              |
| Streamlit    | Web interface                        |
| LangChain    | RAG pipeline and document processing |
| Hugging Face | Local embedding generation           |
| ChromaDB     | Vector storage and similarity search |
| Groq         | Large language model inference       |
| PyPDF2       | PDF text extraction                  |

## Installation

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/research2code.git
cd research2code
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The application will be available through the local Streamlit server.

## API Key

Research2Code requires a Groq API key for language model inference.

The API key can be provided through the application's configuration interface.

For production deployments, API keys should be stored using Streamlit Secrets or environment variables and should never be committed to the repository.

## Project Structure

```text
research2code/
│
├── app.py
├── requirements.txt
├── run.ps1
├── .gitignore
└── README.md
```

