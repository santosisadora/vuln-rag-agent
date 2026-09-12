import os
import asyncio
from dotenv import load_dotenv
from langsmith import Client, aevaluate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.agent.graph import build_vulnerability_graph

load_dotenv()
client = Client()

# Global reference for the compiled LangGraph app so the evaluator can access it
agent_app = None


async def predict_ticket(inputs: dict) -> dict:
    """Wrapper to run the graph and automatically bypass the HITL pause for testing."""
    global agent_app
    config = {"configurable": {"thread_id": f"eval-{os.urandom(4).hex()}"}}

    # 1. Run until the HITL breakpoint using astream
    async for _ in agent_app.astream({"messages": [HumanMessage(content=inputs["query"])]}, config=config):
        pass

    # 2. Automatically resume (simulating human approval) to get the final ticket
    final_output = "No ticket generated"
    async for event in agent_app.astream(None, config=config):
        if "formatter" in event:
            # Safely handle the output depending on how your formatter state is structured
            try:
                final_output = event["formatter"].get("final_ticket", "No ticket found").model_dump_json()
            except AttributeError:
                # Fallback if the output is not a Pydantic model
                final_output = str(event["formatter"])

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
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

    actual = run.outputs.get("actual_ticket", "")
    expected = example.outputs

    prompt = (
        f"You are a QA Auditor. Review this JSON ticket:\n{actual}\n\n"
        f"Did the agent assign exactly Severity: {expected['expected_severity']} "
        f"and an SLA deadline of {expected['expected_sla_days']} days? "
        "Respond strictly with 'PASS' or 'FAIL'."
    )

    # Force cast to string before calling strip() to avoid the list attribute error
    result = str(llm.invoke(prompt).content).strip()
    score = 1 if "PASS" in result else 0
    return {"key": "sla_compliance_score", "score": score}


async def run_evaluations():
    """Asynchronous wrapper to run the LangSmith evaluation."""
    global agent_app
    dataset_name = create_golden_dataset()
    print("\nStarting automated LangSmith evaluation...")

    # Boot up the PostgreSQL pool specifically for testing
    DB_URI = os.getenv("DATABASE_URL")

    async with AsyncConnectionPool(
            conninfo=DB_URI,
            max_size=5,
            kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        # Initialize the checkpointer and compile the app for the test suite
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        agent_app = build_vulnerability_graph(memory=checkpointer)

        # Run the asynchronous evaluation pipeline
        await aevaluate(
            predict_ticket,
            data=dataset_name,
            evaluators=[qa_evaluator],
            experiment_prefix="vuln-agent-regression-",
            metadata={"environment": "ci-cd-test"}
        )
        print("\nEvaluation complete! Check your LangSmith dashboard.")


if __name__ == "__main__":
    # Execute the async evaluation workflow
    asyncio.run(run_evaluations())