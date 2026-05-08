from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import os
import google.generativeai as genai
genai.configure(api_key="AIzaSyCBC_19TeVm4taYld4zAn8WIPIu-GvVU3I")

model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------- MODEL ----------------

model_name = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ---------------- LOAD PDF ----------------

def load_bank_documents():

    file_path = "data/RBI - Bank documents.pdf"

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents

# ---------------- SPLIT TEXT ----------------

def split_bank_text(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(documents)

    return split_docs

# ---------------- VECTOR DB ----------------

def create_bank_vector_db(split_docs):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

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

# ---------------- INITIALIZE ----------------

def initialize_rag_system():

    documents = load_bank_documents()

    split_docs = split_bank_text(documents)

    db = create_bank_vector_db(split_docs)

    return db

# ---------------- GENERATE ANSWER ----------------

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

# ---------------- SEARCH FUNCTION ----------------

def search_bank_answer(db, query):

    docs = db.max_marginal_relevance_search(
        query,
        k=2,
        fetch_k=10
    )

    if not docs:
        return "I could not find the answer in the RBI documents."

    # Select best matching chunk
    best_doc = docs[0]

    for doc in docs:
        if query.lower() in doc.page_content.lower():
            best_doc = doc
            break

    answer = best_doc.page_content

    # Clean formatting
    answer = " ".join(answer.split())

    # Limit answer length
    answer = answer[:500]

    return answer