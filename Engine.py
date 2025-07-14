from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

import fitz  # PyMuPDF

# Load resume PDF and return text
def load_resume_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Create retrieval QA chain using LangChain and Ollama
def build_qa_chain(text):
    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Use HuggingFace Embeddings (offline and free)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Use Ollama LLM (offline)
    llm = Ollama(model="phi3:mini")
  # or "phi3", "llama2" etc.
    retriever = vectorstore.as_retriever()

    # Create retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
