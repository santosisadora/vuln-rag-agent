import pandas as pd
import requests
import json
import mlflow
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import time

load_dotenv()
# 1. Load the Evaluation Dataset
eval_df = pd.read_csv("eval_data.csv")


def predict(inputs):
    """Hits the LIVE FastAPI streaming endpoint over HTTP."""
    if isinstance(inputs, pd.DataFrame):
        queries = inputs["inputs"].tolist()
    elif isinstance(inputs, pd.Series):
        queries = inputs.tolist()
    else:
        queries = inputs

    predictions = []

    for query in queries:
        full_response = ""
        try:
            response = requests.post(
                "http://localhost:8000/triage/stream",
                json={"query": query},
                stream=True
            )

            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8").replace("data: ", ""))
                    if data["type"] == "token":
                        raw_content = data["content"]
                        if isinstance(raw_content, str):
                            full_response += raw_content
                        elif isinstance(raw_content, list):
                            for item in raw_content:
                                if isinstance(item, dict) and "text" in item:
                                    full_response += item["text"]
                                elif isinstance(item, str):
                                    full_response += item
        except Exception as e:
            full_response = f"Error connecting to API: {e}"

        predictions.append(full_response)
    return predictions


# 2. Configure the Gemini Judge
judge_llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash")



# Metric 1: Hallucination / Faithfulness
def eval_hallucination(predictions, targets):
    scores = []
    for i in range(len(predictions)):
        pred = predictions.iloc[i] if hasattr(predictions, "iloc") else predictions[i]
        context = eval_df.iloc[i]["context"]

        prompt = f"""
        You are an auditor checking for AI hallucinations.
        Context: {context}
        Agent Answer: {pred}

        Is the agent's answer strictly supported by the context without inventing false facts or CVEs?
        Answer with only 'YES' or 'NO'.
        """
        # Force the output to a string to prevent list crashes
        raw_content = judge_llm.invoke([HumanMessage(content=prompt)]).content
        res = str(raw_content).strip().upper()
        scores.append(1.0 if "YES" in res else 0.0)

        # 10-second pause to safely respect the 10 RPM limit
        time.sleep(10)

    return mlflow.metrics.MetricValue(
        scores=scores,
        aggregate_results={"faithfulness_rate": sum(scores) / len(scores)}
    )


# Metric 2: Context Precision / Answer Correctness
def eval_correctness(predictions, targets):
    scores = []
    for i in range(len(predictions)):
        pred = predictions.iloc[i] if hasattr(predictions, "iloc") else predictions[i]
        target = targets.iloc[i] if hasattr(targets, "iloc") else targets[i]

        prompt = f"""
        You are a security auditor evaluating accuracy.
        Expected Answer (Ground Truth): {target}
        Agent Answer: {pred}

        Does the agent answer correctly match the ground truth?
        Answer with only 'YES' or 'NO'.
        """
        # Force the output to a string to prevent list crashes
        raw_content = judge_llm.invoke([HumanMessage(content=prompt)]).content
        res = str(raw_content).strip().upper()
        scores.append(1.0 if "YES" in res else 0.0)

        # 10-second pause to safely respect the 10 RPM limit
        time.sleep(10)

    return mlflow.metrics.MetricValue(
        scores=scores,
        aggregate_results={"correctness_rate": sum(scores) / len(scores)}
    )


hallucination_metric = mlflow.metrics.make_metric(eval_fn=eval_hallucination, greater_is_better=True,
                                                  name="faithfulness")
correctness_metric = mlflow.metrics.make_metric(eval_fn=eval_correctness, greater_is_better=True,
                                                name="answer_correctness")

# 3. Execute
if __name__ == "__main__":
    print("Starting Live API Evaluation...")
    mlflow.set_experiment("SecOps_Agent_Live_Evals")
    mlflow.autolog(disable=True)

    with mlflow.start_run(run_name="Live_FastAPI_Run"):
        results = mlflow.models.evaluate(
            model=predict,
            data=eval_df,
            model_type="text",
            targets="ground_truth",
            extra_metrics=[hallucination_metric, correctness_metric]
        )

    print("\n✅ Live Evaluation Complete!")
    print(f"Faithfulness Score: {results.metrics.get('faithfulness_faithfulness_rate', 0) * 100}%")
    print(f"Correctness Score: {results.metrics.get('answer_correctness_correctness_rate', 0) * 100}%")