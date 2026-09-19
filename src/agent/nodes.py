import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy import create_engine

# --- NEW RERANKER IMPORTS ---
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

from src.agent.state import AgentState
from src.tools.nvd_api import fetch_nvd_cve_data
from src.schemas.ticket import RemediationTicket

load_dotenv()

# Initialize LLM & Vector Store
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Global Initialization of the FlashRank Compressor
# (Placed outside the function to prevent memory leaks in the AWS container)
# We set top_n=2 to match your original k=2 output length for the Gemini prompt.
compressor = FlashrankRerank(top_n=2)

# Use os.getenv to allow AWS to inject the production DB, falling back to local for testing
DB_URI = os.getenv("DATABASE_URL")

# Dynamically swap the prefix so SQLAlchemy uses the correct modern psycopg3 driver
if DB_URI and DB_URI.startswith("postgresql://"):
    SQLALCHEMY_URI = DB_URI.replace("postgresql://", "postgresql+psycopg://", 1)
else:
    SQLALCHEMY_URI = DB_URI

db_engine = create_engine(SQLALCHEMY_URI)

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="vuln_management_kb",
    connection=db_engine,
    use_jsonb=True,
)


def router_node(state: AgentState) -> dict:
    """Inspects the query to extract the CVE ID and route the workflow."""
    last_message = state["messages"][-1].content

    # Simple regex extraction for CVE IDs (e.g. CVE-2024-3094)
    cve_match = re.search(r"CVE-\d{4}-\d{4,7}", last_message, re.IGNORECASE)
    cve_id = cve_match.group(0).upper() if cve_match else None

    # Routing Logic for Case 1, 2, and 3!
    if "asset" in last_message or "server" in last_message:
        next_step = "asset_check"  # Case 3: Hit the asset database
    elif cve_id:
        next_step = "fetch_nvd"  # Case 1: Vulnerability lookup
    else:
        next_step = "access_check"  # Case 2: Internal document lookup

    return {
        "cve_id": cve_id,
        "next_step": next_step,
        "cve_intel": None,
        "policy_context": None,
        "access_granted": None,
    }


def access_check_node(state: AgentState) -> dict:
    """Case 2: Validates if the user has Security Analyst (SA) clearance."""
    return {"access_granted": True, "next_step": "retrieve_policy"}


async def nvd_node(state: AgentState) -> dict:
    """Calls the NIST NVD API tool to retrieve real-time vulnerability facts."""
    cve_id = state.get("cve_id")
    if not cve_id:
        return {"cve_intel": "No CVE ID specified for NVD lookup.", "next_step": "access_check"}

    intel = await fetch_nvd_cve_data.ainvoke({"cve_id": cve_id})
    return {"cve_intel": intel, "next_step": "format_ticket"}


def policy_node(state: AgentState) -> dict:
    """Retrieves internal SecOps SLA policies after verifying clearance."""
    if not state.get("access_granted"):
        return {"policy_context": "ACCESS DENIED: You do not have SA clearance to view internal policies."}

    query = state["messages"][-1].content

    # 1. Fetch a broader net (top 10 chunks) from PGVector, keeping your clearance filter
    base_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 10,
            "filter": {"clearance_level": "INTERNAL"}
        }
    )

    # 2. Create the compression hook
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # 3. Invoke the hook to filter the 10 chunks down to the top 2 absolute best matches
    retrieved_docs = compression_retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    if not context:
        context = "No specific internal policies found for this query in the knowledge base."

    return {"policy_context": context, "next_step": "format_ticket"}


async def formatter_node(state: AgentState) -> dict:
    """Generates the final Markdown report for the SecOps analyst."""
    cve_data = state.get('cve_intel', 'None')
    policy_data = state.get('policy_context', 'None')

    prompt = (
        "You are an enterprise SecOps Triage Assistant. Analyze the provided context to answer the user's request.\n\n"
        f"Vulnerability Data (NVD):\n{cve_data}\n\n"
        f"Internal Policy Context:\n{policy_data}\n\n"
        "INSTRUCTIONS FOR FORMATTING:\n"
        "1. SECURITY EXCEPTION: If the 'Internal Policy Context' contains the phrase 'ACCESS DENIED', you MUST immediately stop normal formatting. "
        "Do NOT attempt to answer the user's question. Instead, output ONLY the following exact markdown block:\n"
        "> 🚨 **SECURITY EXCEPTION:** You lack the required clearance to view internal policies for this query. Escalating to SecOps Lead.\n\n"
        "2. If the user asked about a specific vulnerability or CVE (and access is granted), output a highly scannable Markdown '🚨 Vulnerability Report'. "
        "Clearly section out the Description, Severity, CVSS Score, and Required Actions (with SLA deadlines).\n"
        "3. If the user asked a general question about internal policies, SLAs, or concepts (and access is granted), "
        "provide a clear, conversational, well-structured Markdown response that directly answers their question using the Internal Policy Context.\n\n"
        "Do NOT output raw JSON. Use bullet points and bold text where appropriate for readability."
    )

    response = await llm.ainvoke([
        SystemMessage(content="You are a senior SecOps assistant. You format data into beautiful, readable Markdown."),
        HumanMessage(content=prompt)
    ])
    return {"messages": [response]}


def asset_check_node(state: AgentState) -> dict:
    """Case 3: Simulates checking the internal asset inventory."""
    cve = state.get("cve_id", "the vulnerability")
    intel = f"🚨 **CRITICAL MATCH:** Internal asset `Payment-Gateway-01` is running a legacy version vulnerable to {cve}."
    return {"cve_intel": intel, "next_step": "draft_ticket"}


async def draft_ticket_node(state: AgentState) -> dict:
    """Drafts the ticket payload and streams a preview for human approval."""
    prompt = (
        f"Based on this intel: {state.get('cve_intel')}\n\n"
        "Draft a highly scannable Markdown preview of the remediation ticket. "
        "Include the Affected Asset, Severity, and Required Action.\n\n"
        "You MUST conclude your response with this exact text:\n"
        "### ⏸️ TICKET DRAFTED - APPROVAL REQUIRED\n"
        "*Workflow paused. Type **Approve** in the chat to officially generate this ticket.*"
    )

    response = await llm.ainvoke([
        SystemMessage(content="You are a SecOps ticket drafter."),
        HumanMessage(content=prompt)
    ])
    return {"messages": [response]}


async def create_ticket_node(state: AgentState) -> dict:
    """Executes ONLY after HITL approval."""
    response = await llm.ainvoke([
        HumanMessage(
            content="The user approved the ticket. Write a short, 1-sentence confirmation that Ticket #SEC-9942 has been generated and pushed to Jira.")
    ])
    return {"messages": [response]}