import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from src.agent.graph import app as agent_app

app = FastAPI(
    title="Enterprise Vulnerability Triage API",
    version="1.0.0",
    description="Agentic RAG pipeline for SecOps vulnerability management."
)

class TriageRequest(BaseModel):
    query: str
    thread_id: str = "default-thread"


@app.get("/health")
async def health_check():
    """Standard Kubernetes-compatible health probe."""
    return {"status": "ok", "service": "vuln-rag-agent"}


async def event_generator(request: TriageRequest):
    """Generates a Server-Sent Event (SSE) stream from LangGraph execution."""
    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = {"messages": [HumanMessage(content=request.query)]}

    # astream_events(version="v2") allows us to listen to the graph in real-time
    async for event in agent_app.astream_events(inputs, config=config, version="v2"):
        kind = event.get("event")
        name = event.get("name", "unknown")

        # 1. Broadcast Tool Executions
        if kind == "on_tool_start":
            payload = {"type": "tool", "content": f"Executing {name}..."}
            yield f"data: {json.dumps(payload)}\n\n"

        # 2. Broadcast Node Transitions
        elif kind == "on_chain_end" and name in ["router", "nvd_agent", "policy_agent", "formatter"]:
            payload = {"type": "node", "content": f"Node [{name}] completed."}
            yield f"data: {json.dumps(payload)}\n\n"

        # 3. Broadcast LLM Token Streaming
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                payload = {"type": "token", "content": chunk}
                yield f"data: {json.dumps(payload)}\n\n"


@app.post("/triage/stream")
async def stream_triage(request: TriageRequest):
    """Initiates the vulnerability triage agent and streams the response."""
    return StreamingResponse(event_generator(request), media_type="text/event-stream")