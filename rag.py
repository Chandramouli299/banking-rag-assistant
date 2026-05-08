from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# -----------------------------
# Load PDF Documents
# -----------------------------
def load_bank_documents():
    file_path = r"C:\Users\chandramouli\Desktop\bank_rag_project\data\RBI - Bank documents.pdf"

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents


# -----------------------------
# Split Text into Chunks
# -----------------------------
def split_bank_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(documents)
    return split_docs


# -----------------------------
# Create / Load Vector Database
# -----------------------------
def create_bank_vector_db(split_docs):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Save FAISS index for faster reuse
    if os.path.exists("faiss_index"):
        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
       )
    else:
        db = FAISS.from_documents(split_docs, embeddings)
        db.save_local("faiss_index")

    return db


# -----------------------------
# Initialize RAG System
# -----------------------------
def initialize_rag_system():
    documents = load_bank_documents()
    split_docs = split_bank_text(documents)

    db = create_bank_vector_db(split_docs)

    return db


# -----------------------------
# Search Function
# -----------------------------
def generate_answer(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=120
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response
# -----------------------------
# Run standalone test
# -----------------------------
if __name__ == "__main__":
    print("Initializing RAG system...")
    db = initialize_rag_system()

    while True:
        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        response = generate_answer(db, query)
        print("\nAnswer:\n", response)