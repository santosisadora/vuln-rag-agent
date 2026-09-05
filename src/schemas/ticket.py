from pydantic import BaseModel, Field

class RemediationTicket(BaseModel):
    cve_id: str = Field(..., description="The ID of the CVE, e.g., CVE-2024-3094")
    severity: str = Field(..., description="Severity level: LOW, MEDIUM, HIGH, CRITICAL")
    cvss_score: float = Field(..., description="The CVSS v3.1 base score")
    description: str = Field(..., description="Brief summary of the vulnerability")
    required_action: str = Field(..., description="Specific remediation steps based on internal SecOps policies")
    sla_deadline_days: int = Field(..., description="Maximum allowed days to remediate based on SLA policies")