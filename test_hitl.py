import json
from langchain_core.messages import HumanMessage
from src.agent.graph import app


def run_interactive_triage():
    # In production, this thread_id would map to a specific user's session or Jira ticket ID
    config = {"configurable": {"thread_id": "SOC-INCIDENT-999"}}

    print("\n[SOC Analyst]: Asking agent to triage a critical vulnerability...")
    initial_input = {
        "messages": [
            HumanMessage(content="Triage CVE-2024-3094. It is on our production bastion host.")
        ]
    }

    # 1. Run the graph. It will execute NVD and Policy nodes, then PAUSE.
    for event in app.stream(initial_input, config=config):
        for key, value in event.items():
            print(f"--> Finished running node: {key}")

    # 2. Check the graph's current state
    snapshot = app.get_state(config)
    print("\n=== SYSTEM PAUSED (HUMAN-IN-THE-LOOP) ===")
    print(f"Pending execution of node: {snapshot.next}")
    print("\n[Pending Data for Review]:")
    print(f"CVE: {snapshot.values.get('cve_id')}")
    print("-----------------------------------------")

    # 3. Await human approval
    user_input = input("\nDo you approve generating the official compliance ticket based on this data? (yes/no): ")

    if user_input.strip().lower() in ["yes", "y"]:
        print("\n[System]: Approval received. Resuming graph execution...")

        # Pass None to resume from the exact state it paused at
        for event in app.stream(None, config=config):
            if "formatter" in event:
                ticket = event["formatter"]["final_ticket"]
                print("\n=== FINAL REMEDIATION TICKET ===")
                print(json.dumps(ticket.model_dump(), indent=2))
    else:
        print("\n[System]: Ticket generation aborted by human analyst.")


if __name__ == "__main__":
    run_interactive_triage()