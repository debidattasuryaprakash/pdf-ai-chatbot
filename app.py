import os
import re
import tempfile
from html import escape

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

load_dotenv()

st.set_page_config(page_title="Multi-PDF AI Chatbot", page_icon="📚", layout="wide")


@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )


def load_uploaded_pdfs(uploaded_files):
    all_documents = []
    temp_paths = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
            temp_paths.append(temp_path)

        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata["source_file"] = uploaded_file.name

        all_documents.extend(documents)

    return all_documents, temp_paths


def build_vectorstore_from_uploaded_pdfs(uploaded_files):
    if not uploaded_files:
        raise ValueError("No PDFs uploaded.")

    documents, temp_paths = load_uploaded_pdfs(uploaded_files)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    for temp_path in temp_paths:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return vectorstore, chunks


def is_multi_document_query(query: str) -> bool:
    normalized_query = query.lower()
    multi_doc_phrases = [
        "all uploaded pdf",
        "all pdf",
        "all documents",
        "all uploaded documents",
        "uploaded pdfs",
        "uploaded documents",
        "across all",
        "compare",
        "summarize all",
    ]
    return any(phrase in normalized_query for phrase in multi_doc_phrases)


def retrieve_context(vectorstore, query: str, active_files, k: int = 4):
    docs = vectorstore.similarity_search(query, k=k)

    if active_files and (is_multi_document_query(query) or len({doc.metadata.get("source_file") for doc in docs}) == 1):
        selected_docs = []
        seen_keys = set()

        for source_file in active_files:
            source_docs = vectorstore.similarity_search(
                query,
                k=2,
                filter={"source_file": source_file},
            )
            for doc in source_docs:
                key = (
                    doc.metadata.get("source_file"),
                    doc.metadata.get("page"),
                    doc.page_content,
                )
                if key not in seen_keys:
                    selected_docs.append(doc)
                    seen_keys.add(key)

        if selected_docs:
            docs = selected_docs

    context = "\n\n".join([doc.page_content for doc in docs])
    return docs, context


def build_numbered_context(docs):
    numbered_sections = []

    for i, doc in enumerate(docs, start=1):
        source_file = doc.metadata.get("source_file", "unknown file")
        page = doc.metadata.get("page", "unknown")
        page_display = page + 1 if isinstance(page, int) else page

        numbered_sections.append(
            f"""[Source {i}]
File: {source_file}
Page: {page_display}
Content:
{doc.page_content}
"""
        )

    return "\n\n".join(numbered_sections)


def generate_answer(llm, docs, question: str):
    numbered_context = build_numbered_context(docs)

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using only the provided sources.

Rules:
1. Cite claims with inline source labels like [Source 1], [Source 2].
2. If a sentence is supported by multiple sources, cite multiple labels, for example: [Source 1][Source 3]
3. Do not invent facts not present in the sources.
4. If the answer is not in the sources, say exactly: "I couldn't find that in the uploaded PDFs."
5. Keep the answer concise and factual.

Sources:
{numbered_context}

Question:
{question}
"""
    response = llm.invoke(prompt)
    return response.content


def normalize_citations(answer: str) -> str:
    answer = re.sub(
        r"\[\s*Source\s+(\d+)\s*\]",
        r"[Source \1]",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(
        r"(?<!\[)Source\s+(\d+)(?!\])",
        r"[Source \1]",
        answer,
        flags=re.IGNORECASE,
    )
    return answer


def split_into_paragraphs(text: str):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []
    return paragraphs


def choose_best_paragraph(query: str, text: str):
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return ""

    query_terms = set(re.findall(r"\w+", query.lower()))
    best_para = paragraphs[0]
    best_score = -1

    for para in paragraphs:
        para_terms = set(re.findall(r"\w+", para.lower()))
        score = len(query_terms.intersection(para_terms))
        if score > best_score:
            best_score = score
            best_para = para

    return best_para


def highlight_text(text: str, query: str):
    safe_text = escape(text)
    query_terms = sorted(set(re.findall(r"\w+", query)), key=len, reverse=True)

    for term in query_terms:
        if len(term) < 3:
            continue
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        safe_text = pattern.sub(
            lambda m: f"<mark>{escape(m.group(0))}</mark>",
            safe_text,
        )

    return safe_text


def answer_found(answer: str) -> bool:
    return "i couldn't find that in the uploaded pdfs." not in answer.lower()


def render_sources(docs, query: str, found_answer: bool):
    expander_title = (
        "Evidence used for this answer"
        if found_answer
        else "Closest matching passages retrieved"
    )

    with st.expander(expander_title, expanded=True):
        for i, src in enumerate(docs, start=1):
            page = src.metadata.get("page", "unknown")
            source_file = src.metadata.get("source_file", "unknown file")
            page_display = page + 1 if isinstance(page, int) else page

            best_paragraph = choose_best_paragraph(query, src.page_content)
            highlighted = highlight_text(best_paragraph, query)

            st.markdown(
                f"### Source {i}\n"
                f"**File:** `{source_file}`  \n"
                f"**Page:** `{page_display}`"
            )

            st.markdown("**Highlighted paragraph match:**")
            st.markdown(
                f"""
<div style="padding: 12px; border-radius: 8px; background-color: #f5f5f5; color: #111827; margin-bottom: 10px; line-height: 1.6;">
{highlighted}
</div>
""",
                unsafe_allow_html=True,
            )

            with st.expander(f"Show full retrieved chunk for Source {i}"):
                st.text(src.page_content)


st.title("📚 Multi-PDF AI Chatbot")
st.caption("Upload one or more PDFs, then ask questions across all of them.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

llm = get_llm()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is not set.")
    st.stop()

with st.sidebar:
    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write("Selected files:")
        for file in uploaded_files:
            st.write(f"- {file.name}")

    if st.button("Process PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Reading PDFs, creating embeddings, and building vector database..."):
                try:
                    vectorstore, chunks = build_vectorstore_from_uploaded_pdfs(uploaded_files)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.vectorstore_ready = True
                    st.session_state.uploaded_file_names = [file.name for file in uploaded_files]
                    st.session_state.messages = []
                    st.success(f"Processed {len(uploaded_files)} PDF(s) into {len(chunks)} chunks.")
                except Exception as e:
                    st.session_state.vectorstore_ready = False
                    st.session_state.vectorstore = None
                    st.error(f"Processing failed: {e}")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("Reset app", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vectorstore_ready = False
        st.session_state.vectorstore = None
        st.session_state.uploaded_file_names = []
        st.rerun()

    st.divider()
    st.subheader("Active PDFs")
    if st.session_state.uploaded_file_names:
        for name in st.session_state.uploaded_file_names:
            st.write(f"- {name}")
    else:
        st.write("No PDFs processed yet.")

if not st.session_state.vectorstore_ready:
    st.info("Upload PDF files in the sidebar and click 'Process PDFs' to begin.")
    st.stop()

vectorstore = st.session_state.vectorstore
if vectorstore is None:
    st.error("Vector store is not available. Please process the PDFs again.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(
                message["sources"],
                message.get("query", ""),
                answer_found(message["content"]),
            )

user_query = st.chat_input("Ask a question about the uploaded PDFs")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching uploaded PDFs and generating answer..."):
            docs, _ = retrieve_context(
                vectorstore,
                user_query,
                st.session_state.uploaded_file_names,
            )
            answer = generate_answer(llm, docs, user_query)
            answer = normalize_citations(answer)

        st.markdown(answer)
        render_sources(docs, user_query, answer_found(answer))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": docs,
            "query": user_query,
        }
    )
