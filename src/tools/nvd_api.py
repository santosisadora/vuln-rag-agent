import httpx
from langchain_core.tools import tool


@tool
def fetch_nvd_cve_data(cve_id: str) -> str:
    """
    Fetches real-time vulnerability details (CVSS score, description, severity)
    from the NIST NVD API 2.0 for a given CVE ID.
    Always use this tool when asked about a specific vulnerability.
    """
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"

    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()

        if not data.get("vulnerabilities"):
            return f"No data found in NVD for {cve_id}."

        cve_data = data["vulnerabilities"][0]["cve"]
        metrics = cve_data.get("metrics", {})

        # Extract CVSS v3.1 data
        cvss_data = metrics.get("cvssMetricV31", [{}])[0].get("cvssData", {})
        score = cvss_data.get("baseScore", "Unknown")
        severity = cvss_data.get("baseSeverity", "Unknown")

        # Extract English description
        description = next(
            (desc["value"] for desc in cve_data.get("descriptions", []) if desc["lang"] == "en"),
            "No description available."
        )

        return (
            f"CVE: {cve_id}\n"
            f"CVSS v3.1 Score: {score}\n"
            f"Severity: {severity}\n"
            f"Description: {description}"
        )
    except httpx.HTTPError as e:
        return f"HTTP Error fetching NVD data for {cve_id}: {str(e)}"
    except Exception as e:
        return f"Unexpected error processing NVD data for {cve_id}: {str(e)}"