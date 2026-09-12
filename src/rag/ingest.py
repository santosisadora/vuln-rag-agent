import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from sqlalchemy import create_engine
from langchain_core.documents import Document

load_dotenv()


def build_vector_store():
    print("Loading internal SecOps policies securely...")

    policy_dir = Path("data/policies")
    docs = []

    for file_path in policy_dir.glob("*.md"):
        # Explicit encoding prevents cross-platform deployment bugs
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 2. Map directly to LangChain Core Document
        doc = Document(
            page_content=content,
            metadata={
                "source": file_path.name,
                "clearance_level": "INTERNAL",
                "doc_type": "sla_policy"
            }
        )
        docs.append(doc)

    # 3. Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print(f"Initializing PGVector with {len(chunks)} document chunks...")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 4. Database connection setup for SQLAlchemy
    DB_URI = os.getenv("DATABASE_URL")
    if DB_URI and DB_URI.startswith("postgresql://"):
        SQLALCHEMY_URI = DB_URI.replace("postgresql://", "postgresql+psycopg://", 1)
    else:
        SQLALCHEMY_URI = DB_URI

    engine = create_engine(SQLALCHEMY_URI)

    # 5. Build and persist to PostgreSQL
    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="vuln_management_kb",
        connection=engine,
        use_jsonb=True,
    )

    print("Success! Vector store persisted to AWS RDS PostgreSQL.")
    return vectorstore


if __name__ == "__main__":
    build_vector_store()