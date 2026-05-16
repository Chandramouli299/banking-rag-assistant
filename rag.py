from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import streamlit as st
import os
from groq import Groq

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


# INITIALIZE MODEL
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

    try:

        response = client.models.generate_content(
            model="llama3-8b-8192",
            contents=prompt
        )

        print("FULL RESPONSE:")
        print(response)

        # SAFE TEXT EXTRACTION
        answer = ""

        if hasattr(response, "candidates"):
            for candidate in response.candidates:

                if hasattr(candidate, "content"):

                    if hasattr(candidate.content, "parts"):

                        for part in candidate.content.parts:

                            if hasattr(part, "text"):

                                answer += part.text

        if answer.strip():
            return answer

        return "No response generated from Gemini."

    except Exception as e:
        error_message = str(e)

        if "429" in error_message:
            return "Gemini API quota exceeded. Please try again later."
        return f"Gemini Error: {error_message}"
# ---------------- SEARCH FUNCTION ----------------
BANKING_KEYWORDS = [
    "bank",
    "rbi",
    "account",
    "loan",
    "credit",
    "debit",
    "kyc"
]

def is_banking_question(query):
    query = query.lower()
    return any(word in query for word in BANKING_KEYWORDS)

def search_bank_answer(db, query):
    if not is_banking_question(query):
        return "Please ask only banking or RBI-related questions."

    docs = db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Context:
    {context}

    Question:
    {query}
    """

    answer = generate_answer(prompt)

    return answer