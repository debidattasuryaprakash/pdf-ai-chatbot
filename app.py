from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

pdf_path = "documents/sample.pdf"

# Load PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector DB
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# Create retriever
retriever = vectorstore.as_retriever()

# Create LLM
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Ask first question
query = input("Ask a question about the PDF: ")
response = qa_chain.run(query)

print("\nAnswer:")
print(response)