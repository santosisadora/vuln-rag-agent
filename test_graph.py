from langchain_core.messages import HumanMessage
from src.agent.graph import app
import json


def run_test():
    test_input = {
        "messages": [
            HumanMessage(
                content="We detected CVE-2024-3094 on a core server. What is our required action and SLA deadline?")
        ]
    }

    print("\n--- Running LangGraph Vulnerability Agent ---")
    result = app.invoke(test_input)

    print("\n--- Processed State Results ---")
    print(f"CVE Identified: {result.get('cve_id')}")
    print("\n[Extracted NVD Intel]:")
    print(result.get("cve_intel"))
    print("\n[Retrieved Policy]:")
    print(result.get("policy_context"))
    print("\n[Final Structured Remediation Ticket]:")
    print(json.dumps(result["final_ticket"].model_dump(), indent=2))


if __name__ == "__main__":
    run_test()