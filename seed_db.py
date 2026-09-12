import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load your Gemini API Key
load_dotenv()


def seed_database():
    print("Initializing embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    print("Connecting to ChromaDB...")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="vuln_management_kb"
    )

    # Fake Enterprise SecOps Policies
    policies = [
        Document(
            page_content=(
                "**SecOps Patching SLA Policy v2.4**\n"
                "- CRITICAL (CVSS 9.0 - 10.0): Must be patched within 24 hours of discovery. Emergency change management procedures apply.\n"
                "- HIGH (CVSS 7.0 - 8.9): Must be patched within 7 days.\n"
                "- MEDIUM (CVSS 4.0 - 6.9): Must be patched within 30 days.\n"
                "- LOW (CVSS 0.1 - 3.9): Must be patched within 90 days during standard maintenance windows."
            ),
            metadata={"clearance_level": "INTERNAL", "source": "sla_policy_doc"}
        ),
        Document(
            page_content=(
                "**Escalation Procedures for Unpatchable Assets**\n"
                "If a legacy asset (e.g., Payment-Gateway-01) cannot be patched due to compatibility issues, "
                "a compensating control must be deployed (such as WAF isolation), and a risk acceptance form "
                "must be signed by the VP of Engineering."
            ),
            metadata={"clearance_level": "INTERNAL", "source": "escalation_policy"}
        ),
        Document(
            page_content=(
                "**General Employee Security Guidelines**\n"
                "All employees must lock their screens when leaving their desks and report phishing emails to IT."
            ),
            metadata={"clearance_level": "PUBLIC", "source": "employee_handbook"}
        )
    ]

    print("Embedding documents and saving to database...")
    # This embeds the text using Gemini and saves the vectors to disk
    vectorstore.add_documents(documents=policies)

    print("✅ Database successfully seeded!")


if __name__ == "__main__":
    seed_database()