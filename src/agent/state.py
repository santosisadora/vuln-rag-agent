from typing import TypedDict, Optional, List
from langchain_core.messages import BaseMessage
from src.schemas.ticket import RemediationTicket

class AgentState(TypedDict):
    # Chat message history
    messages: List[BaseMessage]
    # Extracted or supplied CVE ID
    cve_id: Optional[str]
    # Live data pulled from NIST NVD API
    cve_intel: Optional[str]
    # Internal compliance and SLA policy excerpts
    policy_context: Optional[str]
    # Final structured remediation ticket
    final_ticket: Optional[RemediationTicket]
    # Routing flag indicating next action
    next_step: Optional[str]
    #
    access_granted: bool | None
    asset_check: Optional[bool]