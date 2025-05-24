from langgraph.graph import StateGraph, END, START
# from langgraph.graph.schema import TypedState
from langchain_groq import ChatGroq
from typing import TypedDict, Optional
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from packages.prompts import agent_prompts, task_prompts, analysis_configs

class AgentState(TypedDict):
    analysis_type: str
    tab_type:str
    custom_query: Optional[str]
    vectorstore: any
    results: Optional[dict]
    reports: Optional[dict]

def get_llm():
    return ChatGroq(model="llama3-70b-8192")


def agentic_rag(vectorstore, task: str,custom_query:str):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    llm = get_llm()
    retriever = vectorstore.as_retriever()

    agentic_rag = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | agent_prompts[task]
        | llm
        | StrOutputParser()
    )

    return agentic_rag.invoke(custom_query)

def agentic_task(response:dict|str,task:str,agents:list[str]):
    def format_docs(docs:dict):
        return "\n\n".join(f"{agent}:\n {result}" for agent,result in docs.items())
    
    llm = get_llm()

    task_prompt = task_prompts[task]
    agentic_task = (
        {
            "response": RunnablePassthrough(),
            "agents": ", ".join(agents)
        }
        | task_prompt
        | llm
        | StrOutputParser()
    )

    response_query = response if isinstance(response,str) else format_docs(response)

    return agentic_task.invoke(response_query)



def coordinator(state: AgentState):
    config = analysis_configs.get(state["analysis_type"])
    if not config:
        raise ValueError(f"Invalid analysis type: {state['analysis_type']}")

    state["custom_query"] = state["custom_query"] if state["analysis_type"] == "Custom Query" else analysis_configs["query"]

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

    return {
        "reports": {
            "Contract Analyst": agentic_rag(
                vectorstore=state["vectorstore"], 
                task="contract", 
                custom_query=state["custom_query"]
            )
        }
    }
    # return {"results": {"Contract Analyst": chain.run(state["query"])}}

def run_research(state: AgentState):

    return {
        "results": {
            "Legal Researcher": agentic_rag(
                vectorstore=state["vectorstore"], 
                task="research", 
                custom_query=state["custom_query"]
            )
        }
    }

def run_strategy(state: AgentState):
   

    return {
        "results": {
            "Legal Strategist": agentic_rag(
                vectorstore=state["vectorstore"], 
                task="strategy", 
                custom_query=state["custom_query"])
        }
    }

def detail_analysis(state: AgentState):
    return {
        "reports": {
            "details": agentic_task(
                response=state["results"],
                task="detail",
                agents=analysis_configs[state["analysis_type"]]["agents"]
            )
        }
    }

def summary_analysis(state: AgentState):
    return {
        "reports": {
            "summary": agentic_task(
                response=state["reports"]["details"],
                task="summary",
                agents=analysis_configs[state["analysis_type"]]["agents"]
            )
        }
    }

def recommendation_analysis(state: AgentState):
    return {
        "reports": {
            "recommendation": agentic_task(
                response=state["reports"]["details"],
                task="recommendation",
                agents=analysis_configs[state["analysis_type"]]["agents"]
            )
        }
    }

def combine_results(state: AgentState):

    merged = {}
    for part in state.get("results", {}).values():
        merged.update(part)
    return {"results": merged}

def build_langgraph():
    workflow = StateGraph(AgentState)
    
    # workflow.add_node("router", coordinator)
    workflow.add_node("contract", run_contract)
    workflow.add_node("research", run_research)
    workflow.add_node("strategy", run_strategy)
    # workflow.add_node("combine", combine_results)
    workflow.add_node("detail", detail_analysis)
    workflow.add_node("summary", summary_analysis)
    workflow.add_node("recommendation", recommendation_analysis)

    # workflow.set_entry_point("router")
    workflow.add_conditional_edges(START, lambda state: coordinator(state), {
        "contract": "contract",
        "research": "research",
        "strategy": "strategy"
    })

    # workflow.add_edge("contract", "combine")
    # workflow.add_edge("research", "combine")
    # workflow.add_edge("strategy", "combine")

    workflow.add_edge("contract", "detail")
    workflow.add_edge("research", "detail")
    workflow.add_edge("strategy", "detail")

    workflow.add_edge("detail", "summary")
    workflow.add_edge("detail", "recommendation")

    workflow.add_edge("summary", END)
    workflow.add_edge("recommendation", END)

    app = workflow.compile()

    app.get_graph().draw_mermaid_png(output_file_path="workflow.png")

    

    return workflow.compile()

if __name__ == "__main__":
    build_langgraph()