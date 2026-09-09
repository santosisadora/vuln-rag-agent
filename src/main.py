import os
import json
import traceback
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from src.agent.graph import app as agent_app

# 1. Define the embedding model used to compare question similarity
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 2. Configure the Semantic Cache interceptor globally
set_llm_cache(InMemoryCache())
print("✅ In-Memory Cache Initialized")

# 3. Initialize the rate limiter (tracks by user IP address)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Enterprise Vulnerability Triage API",
    version="1.0.0",
    description="Agentic RAG pipeline for SecOps vulnerability management."
)

# === SECURITY CONFIGURATIONS ===
# Add Rate Limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS Middleware to allow the browser UI to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://isadora-santos-vuln-rag-agent.s3-website-us-east-1.amazonaws.com",
                   "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================

class TriageRequest(BaseModel):
    query: str
    thread_id: str = "default-thread"


@app.get("/health")
async def health_check():
    """Standard Kubernetes-compatible health probe."""
    return {"status": "ok", "service": "vuln-rag-agent"}


async def event_generator(payload: TriageRequest):
    """Generates a Server-Sent Event (SSE) stream from LangGraph execution."""
    config = {"configurable": {"thread_id": payload.thread_id}}

    # Detect HITL approval: resume the interrupted graph instead of starting fresh
    APPROVAL_KEYWORDS = {"approve", "approved", "yes", "confirm"}
    is_approval = payload.query.strip().lower() in APPROVAL_KEYWORDS

    if is_approval:
        # Resume from the interrupt_before=["create_ticket"] checkpoint
        inputs = None
    else:
        inputs = {"messages": [HumanMessage(content=payload.query)]}

    try:
        # astream_events(version="v2") allows us to listen to the graph in real-time
        async for event in agent_app.astream_events(inputs, config=config, version="v2"):
            kind = event.get("event")
            name = event.get("name", "unknown")

            # 1. Broadcast Tool Executions
            if kind == "on_tool_start":
                yield f"data: {json.dumps({'type': 'tool', 'content': f'Executing {name}...'})}\n\n"

            # 2. Broadcast Node Transitions
            elif kind == "on_chain_end" and name in ["router", "nvd_agent", "policy_agent", "formatter"]:
                yield f"data: {json.dumps({'type': 'node', 'content': f'Node [{name}] completed.'})}\n\n"

            # 3. Broadcast LLM Token Streaming
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

        # Signal the client that the stream has completed cleanly
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        # 1. FORCE the exact error trace into the AWS Fargate CloudWatch logs immediately
        print(f"FATAL STREAM ERROR: {repr(e)}", flush=True)
        traceback.print_exc()

        # 2. Yield the error back into the UI chat bubble so it doesn't fail silently
        error_msg = f"\n\n**⚠️ Backend Crash:** {str(e)}\n\n*Check AWS CloudWatch Logs for full traceback.*"
        yield f"data: {json.dumps({'type': 'token', 'content': error_msg})}\n\n"

        # 3. Close the stream cleanly so the frontend JS doesn't hang
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/triage/stream")
@limiter.limit("5/minute")  # Protects your API quota! Max 5 requests per minute per IP.
async def stream_triage(request: Request, payload: TriageRequest):
    """Initiates the vulnerability triage agent and streams the response."""
    return StreamingResponse(event_generator(payload), media_type="text/event-stream")