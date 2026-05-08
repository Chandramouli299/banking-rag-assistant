from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from transformers import pipeline

# Load free LLM
from transformers import pipeline

llm = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=120
)
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
def search_bank_answer(db, query):

    # Retrieve relevant docs
    results = db.similarity_search(query, k=4)

    # Combine text
    context = " ".join([doc.page_content for doc in results])

    # Clean text
    import re
    context = re.sub(r"\n+", " ", context)
    context = re.sub(r"\s+", " ", context)

    # Simple prompt
    prompt = f"""
    Answer this banking question briefly.

    Question: {query}

    Context: {context}

    Answer:
    """

    # Generate response
    response = llm(prompt)

    # Extract text
    answer = response[0]["generated_text"]

    return answer
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

        response = search_bank_answer(db, query)
        print("\nAnswer:\n", response)