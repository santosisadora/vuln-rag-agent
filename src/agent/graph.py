import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.agent.state import AgentState
from src.agent.nodes import router_node, nvd_node, policy_node, formatter_node


def route_next(state: AgentState) -> str:
    """Conditional edge router function."""
    return state.get("next_step", "format_ticket")


def build_vulnerability_graph():
    workflow = StateGraph(AgentState)

    # 1. Add all nodes
    workflow.add_node("router", router_node)
    workflow.add_node("nvd_agent", nvd_node)
    workflow.add_node("policy_agent", policy_node)
    workflow.add_node("formatter", formatter_node)

    # 2. Add edges
    workflow.add_edge(START, "router")

    # Router conditionally routes to NVD, Policy, or Formatter
    workflow.add_conditional_edges(
        "router",
        route_next,
        {
            "fetch_nvd": "nvd_agent",
            "retrieve_policy": "policy_agent",
            "format_ticket": "formatter"
        }
    )

    # Pipeline transitions
    workflow.add_edge("nvd_agent", "policy_agent")
    workflow.add_edge("policy_agent", "formatter")
    workflow.add_edge("formatter", END)

    # 3. Add SQLite Checkpointer for Memory & HITL
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)

    # Compile with memory and interrupt BEFORE the final ticket formatting
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["formatter"]
    )


app = build_vulnerability_graph()