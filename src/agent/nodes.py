import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.agent.state import AgentState
from src.tools.nvd_api import fetch_nvd_cve_data
from src.schemas.ticket import RemediationTicket

load_dotenv()

# Initialize LLM & Vector Store
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="vuln_management_kb"
)


def router_node(state: AgentState) -> dict:
    """Inspects the query to extract the CVE ID and route the workflow."""
    last_message = state["messages"][-1].content

    # Simple regex extraction for CVE IDs (e.g. CVE-2024-3094)
    cve_match = re.search(r"CVE-\d{4}-\d{4,7}", last_message, re.IGNORECASE)
    cve_id = cve_match.group(0).upper() if cve_match else state.get("cve_id")

    # Routing Logic for Case 1, 2, and 3!
    if "asset" in last_message or "server" in last_message:
        next_step = "asset_check"  # Case 3: Hit the asset database
    elif cve_id and not state.get("cve_intel"):
        next_step = "fetch_nvd"  # Case 1: Vulnerability lookup
    elif not state.get("policy_context"):
        next_step = "access_check"  # Case 2: Internal document lookup
    else:
        next_step = "format_ticket"

    return {"cve_id": cve_id, "next_step": next_step}


def access_check_node(state: AgentState) -> dict:
    """Case 2: Validates if the user has Security Analyst (SA) clearance."""
    # In a production app, you would check JWT tokens or headers here.
    # Since we know they are logged in as an SA, we grant immediate access.
    return {"access_granted": True, "next_step": "retrieve_policy"}


def nvd_node(state: AgentState) -> dict:
    """Calls the NIST NVD API tool to retrieve real-time vulnerability facts."""
    cve_id = state.get("cve_id")
    if not cve_id:
        return {"cve_intel": "No CVE ID specified for NVD lookup.", "next_step": "retrieve_policy"}

    intel = fetch_nvd_cve_data.invoke({"cve_id": cve_id})
    return {"cve_intel": intel, "next_step": "retrieve_policy"}


def policy_node(state: AgentState) -> dict:
    """Retrieves internal SecOps SLA policies after verifying clearance."""
    # 1. Enforce the Access Check
    if not state.get("access_granted"):
        return {"policy_context": "ACCESS DENIED: You do not have SA clearance to view internal policies."}

    # 2. Grab the user's actual question to search the vector store
    query = state["messages"][-1].content

    # 3. Enterprise Metadata Filter (RBAC simulation)
    retrieved_docs = vectorstore.similarity_search(
        query,
        k=2,
        filter={"clearance_level": "INTERNAL"}
    )

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    if not context:
        context = "No specific internal policies found for this query in the knowledge base."

    return {"policy_context": context, "next_step": "format_ticket"}


def formatter_node(state: AgentState) -> dict:
    """Generates the final Markdown report for the SecOps analyst."""
    prompt = (
        "You are an enterprise SecOps Triage Assistant. Analyze the vulnerability data and internal policies.\n"
        f"Vulnerability Data (NVD):\n{state.get('cve_intel')}\n\n"
        f"Internal Policy Context:\n{state.get('policy_context')}\n\n"
        "Provide a clear, highly scannable Markdown report for a security analyst. "
        "Use headers (e.g., '### 🚨 Vulnerability Report'), bullet points, and bold text. "
        "Clearly section out the Description, Severity, CVSS Score, and Required Actions (with SLA deadlines). "
        "Do NOT output raw JSON."
    )

    response = llm.invoke([
        SystemMessage(content="You are a senior SecOps assistant. You format data into beautiful, readable Markdown."),
        HumanMessage(content=prompt)
    ])

    return {"messages": [response]}


def asset_check_node(state: AgentState) -> dict:
    """Case 3: Simulates checking the internal asset inventory."""
    cve = state.get("cve_id", "the vulnerability")
    intel = f"🚨 **CRITICAL MATCH:** Internal asset `Payment-Gateway-01` is running a legacy version vulnerable to {cve}."
    return {"cve_intel": intel, "next_step": "draft_ticket"}


def draft_ticket_node(state: AgentState) -> dict:
    """Drafts the ticket payload and streams a preview for human approval."""
    prompt = (
        f"Based on this intel: {state.get('cve_intel')}\n\n"
        "Draft a highly scannable Markdown preview of the remediation ticket. "
        "Include the Affected Asset, Severity, and Required Action.\n\n"
        "You MUST conclude your response with this exact text:\n"
        "### ⏸️ TICKET DRAFTED - APPROVAL REQUIRED\n"
        "*Workflow paused. Type **Approve** in the chat to officially generate this ticket.*"
    )

    # Use the standard LLM so it streams the beautiful Markdown to the UI!
    response = llm.invoke([
        SystemMessage(content="You are a SecOps ticket drafter."),
        HumanMessage(content=prompt)
    ])

    return {"messages": [response]}


def create_ticket_node(state: AgentState) -> dict:
    """Executes ONLY after HITL approval."""
    # We use the LLM to stream a final confirmation message.
    # (In a real app, this is where you would POST the strict JSON to Jira/ServiceNow)
    response = llm.invoke([
        HumanMessage(
            content="The user approved the ticket. Write a short, 1-sentence confirmation that Ticket #SEC-9942 has been generated and pushed to Jira.")
    ])
    return {"messages": [response]}