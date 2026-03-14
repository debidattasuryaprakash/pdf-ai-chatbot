import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

load_dotenv()

st.set_page_config(page_title="PDF AI Chatbot", page_icon="📄", layout="wide")

PDF_PATH = "documents/sample.pdf"
CHROMA_DIR = "chroma_db"


@st.cache_resource
def build_vectorstore(pdf_path: str):
    """Load PDF, split it, embed it, and store it in Chroma."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )


def retrieve_context(vectorstore, query: str, k: int = 4):
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return docs, context


def generate_answer(llm, context: str, question: str):
    prompt = f"""
You are a helpful AI assistant.
Answer the user's question using only the provided PDF context.
If the answer is not in the context, say: "I couldn't find that in the PDF."

Context:
{context}

Question:
{question}
"""
    response = llm.invoke(prompt)
    return response.content


st.title("📄 PDF AI Chatbot")
st.caption("Ask questions about your PDF using RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    st.write(f"PDF file: `{PDF_PATH}`")
    st.write(f"Vector DB: `{CHROMA_DIR}`")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is not set.")
    st.stop()

try:
    vectorstore = build_vectorstore(PDF_PATH)
    llm = get_llm()
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for i, src in enumerate(message["sources"], start=1):
                    page = src.metadata.get("page", "unknown")
                    preview = src.page_content[:300].replace("\n", " ")
                    st.write(f"**Chunk {i} | Page {page + 1 if isinstance(page, int) else page}**")
                    st.write(preview + "...")

user_query = st.chat_input("Ask a question about the PDF")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF and generating answer..."):
            docs, context = retrieve_context(vectorstore, user_query)
            answer = generate_answer(llm, context, user_query)

        st.markdown(answer)

        with st.expander("Sources"):
            for i, src in enumerate(docs, start=1):
                page = src.metadata.get("page", "unknown")
                preview = src.page_content[:300].replace("\n", " ")
                st.write(f"**Chunk {i} | Page {page + 1 if isinstance(page, int) else page}**")
                st.write(preview + "...")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": docs,
        }
    )