import os
import json
import traceback
import subprocess
import secrets
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.agent.graph import build_vulnerability_graph

# Global variable to hold our compiled app
agent_app = None

# --- BASIC AUTHENTICATION SETUP ---
security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "recruiter")
    correct_password = secrets.compare_digest(credentials.password, "hireme")

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# --- AUTOMATIC POSTGRESQL DB CONNECTION & REBUILD ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executes startup and shutdown logic for the API."""
    print("🚀 Container Starting: Connecting to PostgreSQL...")

    # Uses the environment variable in AWS, or localhost for local testing
    DB_URI = os.getenv("DATABASE_URL")

    # 1. Create a connection pool that stays open while the server runs
    async with AsyncConnectionPool(
            conninfo=DB_URI,
            max_size=20,
            kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:

        # 2. Initialize the Postgres Checkpointer with our pool
        checkpointer = AsyncPostgresSaver(pool)

        # 3. Create the LangGraph state tables in the database if they don't exist yet!
        await checkpointer.setup()

        # 4. Compile our LangGraph app using the new checkpointer
        global agent_app
        agent_app = build_vulnerability_graph(memory=checkpointer)

        print("✅ PostgreSQL Checkpointer initialized!")

        # Automatically run your ingest script to parse the Markdown SLA policies
        # and load them into the fresh PGVector database on boot.
        try:
            subprocess.run(["python", "-m", "src.rag.ingest"], check=True)
            print("✅ Data ingested into PGVector successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Failed to ingest data into PGVector on startup: {e}")

        yield  # The FastAPI server runs while yielding

    print("🛑 Container Shutting Down. Closing DB pool.")


# -----------------------------------

# 1. Define the embedding model used to compare question similarity
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 2. Configure the Semantic Cache interceptor globally
set_llm_cache(InMemoryCache())
print("✅ In-Memory Cache Initialized")

# 3. Initialize the rate limiter (tracks by user IP address)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Enterprise Vulnerability Triage API",
    version="1.0.0",
    description="Agentic RAG pipeline for SecOps vulnerability management.",
    lifespan=lifespan  # Attach the startup script here
)

# === SECURITY CONFIGURATIONS ===
# Add Rate Limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS Middleware to allow the browser UI to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://isadora-santos-vuln-rag-agent.s3-website-us-east-1.amazonaws.com",
        "https://dvd40fbcl8i24.cloudfront.net",
        "http://localhost:8501"
    ],
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
    # We purposefully leave this endpoint UNPROTECTED so the AWS Load Balancer can verify the container is alive.
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
        # Switch from astream_events to standard astream to grab node outputs reliably
        async for output in agent_app.astream(inputs, config=config, stream_mode="updates"):

            # Loop through each node that executed in this step
            for node_name, node_state in output.items():

                # 1. Broadcast Node Transitions so UI shows progress
                yield f"data: {json.dumps({'type': 'node', 'content': f'Node [{node_name}] completed.'})}\n\n"

                # 2. Grab the final output from the formatter, draft_ticket, create_ticket, OR conversational_reply node
                if node_name in ["formatter", "draft_ticket", "create_ticket", "conversational_reply"]:
                    try:
                        # Try to extract standard LangChain message content
                        final_text = node_state["messages"][-1].content
                    except (KeyError, IndexError, AttributeError):
                        # Fallback if the create_ticket node returns a custom dictionary instead of standard messages
                        final_text = f"✅ **Action Approved:** Node `{node_name}` completed successfully."

                    # Yield it as a single chunk to guarantee the UI renders it
                    yield f"data: {json.dumps({'type': 'token', 'content': final_text})}\n\n"

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
async def stream_triage(request: Request, payload: TriageRequest, username: str = Depends(verify_credentials)):
    """Initiates the vulnerability triage agent and streams the response. Now locked behind Basic Auth."""
    return StreamingResponse(event_generator(payload), media_type="text/event-stream")