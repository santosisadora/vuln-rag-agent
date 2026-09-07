import sqlite3
from langgraph.graph import StateGraph, START, END


#from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.nodes import router_node, nvd_node, policy_node, formatter_node, access_check_node, asset_check_node, draft_ticket_node, create_ticket_node


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
    workflow.add_node("access_check", access_check_node)

    workflow.add_node("asset_check", asset_check_node)
    workflow.add_node("draft_ticket", draft_ticket_node)
    workflow.add_node("create_ticket", create_ticket_node)

    # 2. Add edges
    workflow.add_edge(START, "router")
    workflow.add_edge("access_check", "policy_agent")

    workflow.add_edge("asset_check", "draft_ticket")
    workflow.add_edge("draft_ticket", "create_ticket")

    # Router conditionally routes to NVD, Policy, or Formatter
    workflow.add_conditional_edges(
        "router",
        route_next,
        {
            "fetch_nvd": "nvd_agent",
            "access_check": "access_check",
            "asset_check": "asset_check",
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
    # memory = SqliteSaver(conn)
    memory = MemorySaver()

    # Compile with memory and interrupt BEFORE the final ticket formatting
    return workflow.compile(
        checkpointer=memory,
        # it only pauses if the agent tries to create a ticket
        interrupt_before=["create_ticket"]
    )


app = build_vulnerability_graph()