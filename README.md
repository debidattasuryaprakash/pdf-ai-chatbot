# PDF AI Chatbot (RAG System)

A simple AI chatbot that answers questions from PDF documents using Retrieval Augmented Generation (RAG).

## Tech Stack

- Python
- LangChain
- OpenAI
- Chroma Vector Database
- Streamlit

## Features

- Load PDF documents
- Convert text into embeddings
- Store vectors in Chroma DB
- Retrieve relevant document chunks
- Generate answers using an LLM

## Project Structure

pdf-ai-chatbot
|
|-- app.py
|-- requirements.txt
|-- README.md
`-- documents
    `-- sample.pdf

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set OpenAI API key:

Linux/Mac

```bash
export OPENAI_API_KEY="your_key"
```

Windows

```bash
setx OPENAI_API_KEY "your_key"
```

Run the chatbot:

```bash
python app.py
```

## Example Question

What is the main topic of the document?

## What This Project Teaches

- RAG architecture
- embeddings
- vector search
- LLM applications
