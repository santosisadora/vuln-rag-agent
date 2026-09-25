![CI/CD Pipeline](https://github.com/santosisadora/vuln-rag-agent/actions/workflows/deploy.yml/badge.svg?branch=main)

# SecOps Vulnerability Triage Agent (vuln-rag-agent)

An agentic Retrieval-Augmented Generation (RAG) pipeline designed to automate security operations (SecOps) vulnerability triage. Built with LangGraph and FastAPI, this agent analyzes CVEs against internal policies, determines severity, assigns SLA deadlines, and drafts actionable security tickets with a Human-in-the-Loop (HITL) approval process.

## 🏗️ Architecture Stack

* **Agent Framework:** LangGraph, LangChain
* **LLM:** Google Gemini API
* **Backend:** FastAPI, Python 3.11
* **Database & Vector Store:** PostgreSQL with `pgvector` (AWS RDS)
* **Checkpointer:** `langgraph-checkpoint-postgres` (Async)
* **Frontend:** Static HTML/JS
* **CI/CD & Evaluation:** GitHub Actions, LangSmith
* **Cloud Infrastructure:** AWS ECS (Fargate), AWS ECR, AWS S3

## ✨ Key Features

* **Automated CVE Triage:** Ingests vulnerability reports and assesses them against cloud assets and production environments.
* **Policy RAG:** Retrieves internal security policies (via PGVector) to ensure accurate severity scoring and SLA assignment.
* **Human-in-the-Loop (HITL):** Pauses execution before finalizing tickets, requiring human approval before ticket generation.
* **Regression Tested:** Automated CI/CD pipeline runs deterministic LLM-as-a-judge evaluations using LangSmith before every deployment.
* **Fully Cloud Native:** Containerized backend running on AWS ECS Fargate, backed by AWS RDS PostgreSQL, with a decoupled static frontend on AWS S3.

## 🚀 Deployment Pipeline

This repository utilizes GitHub Actions (`deploy.yml`) for automated CI/CD.

1. **Test & Evaluate:** Triggers `src/evaluate.py`.
2. **LangSmith QA:** Runs an LLM-as-a-judge regression test against the `SecOps-Vulnerability-Golden-Set` dataset to ensure SLA compliance accuracy. Uses the live AWS RDS PostgreSQL checkpointer.
3. **Build & Push:** Containerizes the FastAPI backend and pushes the image to AWS ECR.
4. **Deploy:** Triggers an AWS ECS Fargate service update with the new container image.

## ⚙️ Environment Variables

To run this project locally or in CI/CD, the following secrets must be configured:

```env
LANGCHAIN_API_KEY=your_langsmith_api_key
GOOGLE_API_KEY=your_gemini_api_key
DATABASE_URL=postgresql://postgres:<PASSWORD>@<YOUR_RDS_ENDPOINT>:5432/vulnrdb
LANGCHAIN_TRACING_V2=true
