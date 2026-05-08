from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import os

# ---------------- MODEL ----------------

model_name = "google/flan-t5-base"

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
        chunk_size=700,
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

    # Retrieve top matching chunks
    docs = db.similarity_search(query, k=3)

    # Combine context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Better instruction prompt
    prompt = f"""
You are an RBI Banking Assistant.

Read the context carefully and answer the question accurately.

If the exact answer is not available, say:
"I could not find the answer clearly in the RBI documents."

Context:
{context}

Question:
{query}

Answer in simple and complete sentences:
"""

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    # Generate answer
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.1,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer