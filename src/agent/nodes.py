import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.state import AgentState
from src.tools.nvd_api import fetch_nvd_cve_data
from src.schemas.ticket import RemediationTicket

load_dotenv()

# Initialize LLM & Vector Store
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
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

    # Decide next step based on what data we already have
    if cve_id and not state.get("cve_intel"):
        next_step = "fetch_nvd"
    elif not state.get("policy_context"):
        next_step = "retrieve_policy"
    else:
        next_step = "format_ticket"

    return {"cve_id": cve_id, "next_step": next_step}


def nvd_node(state: AgentState) -> dict:
    """Calls the NIST NVD API tool to retrieve real-time vulnerability facts."""
    cve_id = state.get("cve_id")
    if not cve_id:
        return {"cve_intel": "No CVE ID specified for NVD lookup.", "next_step": "retrieve_policy"}

    intel = fetch_nvd_cve_data.invoke({"cve_id": cve_id})
    return {"cve_intel": intel, "next_step": "retrieve_policy"}


def policy_node(state: AgentState) -> dict:
    """Retrieves relevant internal SecOps SLA policies with clearance metadata filtering."""
    query = f"remediation timeline SLA for {state.get('cve_intel', 'vulnerabilities')}"

    # Enterprise Metadata Filter (RBAC simulation)
    retrieved_docs = vectorstore.similarity_search(
        query,
        k=2,
        filter={"clearance_level": "INTERNAL"}
    )

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return {"policy_context": context, "next_step": "format_ticket"}


def formatter_node(state: AgentState) -> dict:
    """Generates the final compliance ticket enforced by Pydantic structured output."""
    structured_llm = llm.with_structured_output(RemediationTicket)

    prompt = (
        "You are an enterprise SecOps Triage Assistant. Analyze the vulnerability data and internal policies.\n"
        f"Vulnerability Data (NVD):\n{state.get('cve_intel')}\n\n"
        f"Internal Policy Context:\n{state.get('policy_context')}\n\n"
        "Generate a strictly-typed remediation ticket respecting internal SLAs."
    )

    ticket = structured_llm.invoke([
        SystemMessage(content="You generate verified SecOps remediation tickets."),
        HumanMessage(content=prompt)
    ])

    return {"final_ticket": ticket}