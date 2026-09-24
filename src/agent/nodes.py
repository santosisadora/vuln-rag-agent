import os
import re
import requests
import json
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from sqlalchemy import create_engine

# --- NEW RERANKER IMPORTS ---
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.runnables.config import RunnableConfig

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
    """Inspects the query to extract the CVE ID and route the workflow using a strict SecOps whitelist."""
    last_message = state["messages"][-1].content
    msg_lower = last_message.lower().strip()

    # Simple regex extraction for CVE IDs (e.g. CVE-2024-3094)
    cve_match = re.search(r"CVE-\d{4}-\d{4,7}", last_message, re.IGNORECASE)
    cve_id = cve_match.group(0).upper() if cve_match else None

    asset_keywords = [
        "asset", "assets", "inventory", "server", "servers",
        "host", "hosts", "affected", "impacted", "gateway"
    ]
    policy_keywords = [
        "policy", "policies", "sla", "remediation", "timeline", "schedule",
        "critical", "high", "medium", "low", "cvss", "severity", "tier",
        "compliance", "standard", "patch", "patching", "escalation",
        "deadline", "vulnerability", "vulnerabilities", "triage", "secops"
    ]

    # Whitelist Routing Logic for Case 1, 2, 3, and Small Talk / Off-Topic Guardrail
    if any(k in msg_lower for k in asset_keywords):
        next_step = "asset_check"  # Case 3: Hit the asset database
    elif cve_id:
        next_step = "fetch_nvd"  # Case 1: Vulnerability lookup
    elif any(k in msg_lower for k in policy_keywords):
        next_step = "access_check"  # Case 2: Internal document lookup
    else:
        next_step = "conversational_reply"  # Catch greetings and off-topic queries before hitting the DB

    return {
        "cve_id": cve_id,
        "next_step": next_step,
        "cve_intel": None,
        "policy_context": None,
        "access_granted": None,
    }


async def conversational_node(state: AgentState) -> dict:
    """Handles greetings, capability inquiries, and off-topic guardrails without querying PGVector."""
    user_message = state["messages"][-1].content if state.get("messages") else "Hello"

    system_prompt = (
        "You are an enterprise SecOps Vulnerability Management Agent built by Isadora Santos.\n"
        "Follow these strict response rules based on the user's message:\n"
        "1. IF GREETING OR CAPABILITY QUESTION: Introduce yourself briefly, stating you can summarize CVEs via live NIST NVD lookups, "
        "check internal asset exposure, retrieve corporate remediation SLA policies, and autonomously draft Jira tickets.\n"
        "2. IF OFF-TOPIC (e.g., weather, general trivia, non-security questions): Politely decline to answer and state that as a "
        "dedicated SecOps Vulnerability Triage Agent, your operational scope is strictly restricted to vulnerability intelligence, "
        "internal asset exposure, and corporate remediation policies.\n"
        "3. CRITICAL REQUIREMENT FOR EVERY RESPONSE: Always conclude by explicitly stating that because this environment is running "
        "as a portfolio demonstration, the user has been granted temporary 'Security Analyst' clearance to freely explore all "
        "restricted features, and invite them to enter a CVE ID, asset query, or SLA policy question."
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])
    return {"messages": [response]}


def access_check_node(state: AgentState, config: RunnableConfig) -> dict:
    """Case 2: Validates if the user has Security Analyst (SA) clearance securely via FastAPI config."""
    access_granted = config.get("configurable", {}).get("access_granted", False)
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
        "3. You MUST use color-coded circle emojis to represent the Severity level (🔴 Critical, 🟠 High, 🟡 Medium, 🟢 Low).\n"
        "4. If the user asked a general question about internal policies, SLAs, or concepts (and access is granted), "
        "provide a clear, conversational, well-structured Markdown response that directly answers their question using the Internal Policy Context.\n"
        "5. DEMO DISCLAIMER: Conclude your response with this exact italicized disclaimer at the very bottom:\n"
        "'*Note: Internal SLA policies are visible because Portfolio Demo Mode is active.*'\n\n"
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
    """Explains affected assets to the user first, then drafts the remediation ticket for human approval."""
    user_query = state["messages"][-1].content if state.get("messages") else ""
    cve_intel = state.get("cve_intel", "None")
    policy_context = state.get("policy_context", "None")

    prompt = (
        f"User Question: {user_query}\n\n"
        f"Vulnerability & Asset Intelligence:\n{cve_intel}\n\n"
        f"Internal Policy Context:\n{policy_context}\n\n"
        "You MUST structure your response in the following exact two-part sequence:\n\n"
        "PART 1 — CONVERSATIONAL ASSET IMPACT EXPLANATION:\n"
        "Start with a header '### 🔍 Asset Impact Analysis'.\n"
        "Directly answer the user's question in clear, conversational prose before showing any ticket. "
        "Explicitly name the affected internal asset(s) found in the inventory (e.g., `Payment-Gateway-01`), "
        "explain what software/component on that asset is vulnerable to the CVE, its severity/exposure risk, "
        "and state: 'Because an active internal asset is impacted by this vulnerability, I have automatically "
        "prepared the following remediation ticket draft for your review and approval.'\n\n"
        "PART 2 — REMEDIATION TICKET DRAFT:\n"
        "Add a horizontal divider (`---`) followed by the header '### 🛡️ Remediation Ticket Draft'.\n"
        "Present a clean Markdown table and description including the Ticket ID, Title, Affected Asset, "
        "Vulnerability, Severity (using color-coded circles 🔴 🟠 🟡 🟢), and Required Action based on SLA policy.\n\n"
        "DEMO DISCLAIMER: Conclude the ticket description with this exact italicized line:\n"
        "'*Note: This automated ticket was generated via Portfolio Demo Mode.*'\n\n"
        "You MUST conclude your entire response with this exact text block:\n"
        "### ⏸️ TICKET DRAFTED - APPROVAL REQUIRED\n"
        "*Workflow paused. Type **Approve** in the chat to officially generate this ticket.*"
    )

    response = await llm.ainvoke([
        SystemMessage(content="You are an enterprise SecOps analyst who clearly explains asset exposure before presenting a ticket draft."),
        HumanMessage(content=prompt)
    ])
    return {"messages": [response]}


async def create_ticket_node(state: AgentState) -> dict:
    """Executes ONLY after HITL approval and pushes a detailed Jira ticket."""

    # Grab the markdown ticket draft from the previous node and strip the HITL warning text
    draft_content = state["messages"][-2].content if len(state["messages"]) > 1 else ""
    clean_draft = draft_content.split("### ⏸️")[0].strip()

    # Fix the "None" title by providing a dynamic fallback
    cve_id = state.get('cve_id')
    summary_title = f"Remediate Vulnerability: {cve_id}" if cve_id else "Remediate Vulnerability: Internal Asset Risk"

    # Build a highly detailed, enterprise-grade description using Jira Wiki markup
    detailed_description = (
        "h2. Automated SecOps Triage Report\n"
        "This ticket was generated autonomously by the Vulnerability RAG Agent.\n\n"
        "h3. Triage Summary\n"
        f"{clean_draft}\n\n"
        "h3. Internal SLA & Policy Context\n"
        f"{state.get('policy_context', 'No specific internal policy matched.')}\n\n"
        "h3. Vulnerability Intelligence\n"
        f"{state.get('cve_intel', 'No additional intelligence gathered.')}"
    )

    url = f"https://{os.environ['JIRA_DOMAIN']}/rest/api/2/issue"
    auth = HTTPBasicAuth(os.environ["JIRA_EMAIL"], os.environ["JIRA_API_TOKEN"])
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    payload = json.dumps({
        "fields": {
            "project": {"key": "KAN"},
            "summary": summary_title,
            "description": detailed_description,
            "issuetype": {"name": "Task"}
        }
    })

    response = requests.post(url, data=payload, headers=headers, auth=auth)

    if response.status_code == 201:
        issue_key = response.json().get("key")
        ticket_url = f"https://{os.environ['JIRA_DOMAIN']}/browse/{issue_key}"

        # Using raw HTML to force the link to open in a new tab
        success_msg = f"✅ **Ticket successfully created!**\n\nView it in Jira here: <a href='{ticket_url}' target='_blank'>{issue_key}</a>"

        return {"messages": [AIMessage(content=success_msg)]}
    else:
        error_msg = f"❌ **Failed to create Jira ticket:**\n```json\n{response.text}\n```"
        return {"messages": [AIMessage(content=error_msg)]}