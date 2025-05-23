from langgraph.graph import StateGraph, END
# from langgraph.graph.schema import TypedState
from langchain_groq import ChatGroq
from typing import TypedDict, Optional
from packages.prompts import analysis_configs, make_chain

class AgentState(TypedDict):
    tab_type:str
    analysis_type: str
    custom_query: Optional[str]
    vectorstore: any
    results: Optional[dict]

def get_llm():
    return ChatGroq(groq_api_key="your-key", model="mixtral-8x7b-32768")


def coordinator(state: AgentState):
    config = analysis_configs.get(state["analysis_type"])
    if not config:
        raise ValueError(f"Invalid analysis type: {state['analysis_type']}")

    state["query"] = state["custom_query"] if state["analysis_type"] == "Custom Query" else config["query"]

    agent_routes = []
    for agent in config["agents"]:
        if agent == "Contract Analyst":
            agent_routes.append("contract")
        elif agent == "Legal Researcher":
            agent_routes.append("research")
        elif agent == "Legal Strategist":
            agent_routes.append("strategy")
    return agent_routes


def run_contract(state: AgentState):
    chain = make_chain(state["vectorstore"], "contract")
    return {"results": {"Contract Analyst": chain.run(state["query"])}}

def run_research(state: AgentState):
    chain = make_chain(state["vectorstore"], "research")
    return {"results": {"Legal Researcher": chain.run(state["query"])}}

def run_strategy(state: AgentState):
    chain = make_chain(state["vectorstore"], "strategy")
    return {"results": {"Legal Strategist": chain.run(state["query"])}}

def combine_results(state: AgentState):
    merged = {}
    for part in state.get("results", {}).values():
        merged.update(part)
    return {"results": merged}

def build_langgraph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("router", coordinator)
    workflow.add_node("contract", run_contract)
    workflow.add_node("research", run_research)
    workflow.add_node("strategy", run_strategy)
    workflow.add_node("combine", combine_results)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", lambda state: coordinator(state), {
        "contract": "contract",
        "research": "research",
        "strategy": "strategy"
    })

    workflow.add_edge("contract", "combine")
    workflow.add_edge("research", "combine")
    workflow.add_edge("strategy", "combine")
    workflow.add_edge("combine", END)

    return workflow.compile()
