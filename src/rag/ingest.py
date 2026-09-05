
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Using the import below in a production environment would cause DeprecationWarning errors and cause headaches
# from langchain_community.document_loaders import TextLoader, DirectoryLoader
# Instead, we use Modern LangChain imports (No community package)
from langchain_core.documents import Document

load_dotenv()

CHROMA_DIR = "./chroma_db"


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

    print(f"Initializing ChromaDB with {len(chunks)} document chunks...")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 4. Build and persist (Notice 'embedding' is singular!)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="vuln_management_kb"
    )

    print(f"Success! Vector store persisted to {CHROMA_DIR}.")
    return vectorstore


if __name__ == "__main__":
    build_vector_store()