import os
from dotenv import load_dotenv
from langsmith import Client, evaluate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agent.graph import app
from langchain_core.messages import HumanMessage

load_dotenv()
client = Client()


def predict_ticket(inputs: dict) -> dict:
    """Wrapper to run the graph and automatically bypass the HITL pause for testing."""
    config = {"configurable": {"thread_id": f"eval-{os.urandom(4).hex()}"}}

    # 1. Run until the HITL breakpoint
    for _ in app.stream({"messages": [HumanMessage(content=inputs["query"])]}, config=config):
        pass

    # 2. Automatically resume (simulating human approval) to get the final ticket
    final_output = "No ticket generated"
    for event in app.stream(None, config=config):
        if "formatter" in event:
            final_output = event["formatter"]["final_ticket"].model_dump_json()

    return {"actual_ticket": final_output}


def create_golden_dataset():
    """Creates a deterministic dataset in LangSmith for regression testing."""
    dataset_name = "SecOps-Vulnerability-Golden-Set"

    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Baseline CVEs to ensure the agent respects SLA timelines."
        )
        client.create_examples(
            inputs=[
                {"query": "Triage CVE-2024-3094. It is on our production bastion host."},
                {"query": "We have CVE-2023-38606 on our payment gateway."}
            ],
            outputs=[
                {"expected_severity": "CRITICAL", "expected_sla_days": 2},  # 48 hours
                {"expected_severity": "HIGH", "expected_sla_days": 14}
            ],
            dataset_id=dataset.id,
        )
        print(f"Created dataset: {dataset_name}")
    return dataset_name


def qa_evaluator(run, example) -> dict:
    """An LLM-as-a-judge evaluator that grades the generated ticket."""
    llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0)

    actual = run.outputs.get("actual_ticket", "")
    expected = example.outputs

    prompt = (
        f"You are a QA Auditor. Review this JSON ticket:\n{actual}\n\n"
        f"Did the agent assign exactly Severity: {expected['expected_severity']} "
        f"and an SLA deadline of {expected['expected_sla_days']} days? "
        "Respond strictly with 'PASS' or 'FAIL'."
    )

    result = llm.invoke(prompt).content.strip()
    score = 1 if "PASS" in result else 0
    return {"key": "sla_compliance_score", "score": score}


if __name__ == "__main__":
    dataset_name = create_golden_dataset()
    print("\nStarting automated LangSmith evaluation...")

    # Run the evaluation pipeline
    evaluate(
        predict_ticket,
        data=dataset_name,
        evaluators=[qa_evaluator],
        experiment_prefix="vuln-agent-regression-",
        metadata={"environment": "ci-cd-test"}
    )
    print("\nEvaluation complete! Check your LangSmith dashboard.")