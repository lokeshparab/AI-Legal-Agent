from langgraph.graph import StateGraph, END, START
from langchain_groq import ChatGroq
from typing import TypedDict, Optional, Annotated, Any, List
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from packages.prompts import agent_prompts, task_prompts, analysis_configs


def merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dictionaries together into one, overwriting keys if necessary.

    Args:
        a (dict): The first dictionary to merge.
        b (dict): The second dictionary to merge.

    Returns:
        dict: The merged dictionary.
    """
    return {**a, **b}


class AgentState(TypedDict):
    analysis_type: str
    custom_query: Optional[str]
    vectorstore: Any
    results: Annotated[dict[str, str], merge_dicts]
    reports: Annotated[dict[str, str], merge_dicts]

def get_llm():
    """
    Returns a ChatGroq model with the LLaMA3-8B-8192 model.

    This model is a large language model that is suitable for generating text based on a given prompt.

    Returns:
        ChatGroq: The LLaMA3-8B-8192 model.
    """
    return ChatGroq(model="llama3-8b-8192")


def agentic_rag(vectorstore, task: str,custom_query:str):
    """
    Use a vectorstore to retrieve relevant documents and then ask a prompt to a large language model.

    Args:
        vectorstore (VectorStore): The vectorstore to use for retrieving documents.
        task (str): The task to use (e.g. "contract", "research", "strategy").
        custom_query (str): The custom query to ask the model.

    Returns:
        str: The result of the model's response.
    """
    def format_docs(docs):
        """
        Take a list of documents and format them into a single string, separated by double newlines.

        Args:
            docs (list[Document]): The list of documents to format.

        Returns:
            str: The formatted string.
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Get LLM and vector retriever
    llm = get_llm()
    retriever = vectorstore.as_retriever()

    # Agentic Rag Chain
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

def agentic_task(response,task:str,agents:list[str]):
    """
    Executes an agentic task by processing a response and generating a text output using specified agents.

    Args:
        response (str or dict): The response data to be processed. Can be a string or a dictionary of agent results.
        task (str): The task identifier which determines the task prompt to be used.
        agents (list[str]): A list of agent names involved in the task.

    Returns:
        str: The resulting output from the large language model after processing the response with the task prompt.
    """

    def format_docs(docs:dict):
        """
        Take a dictionary of documents and format them into a single string, separated by double newlines, 
        with each document identified by its key in the dictionary.

        Args:
            docs (dict): The dictionary of documents to format.

        Returns:
            str: The formatted string.
        """
        return "\n\n".join(f"{agent}:\n {result}" for agent,result in docs.items())
    
    # Get LLM and vector retriever
    llm = get_llm()
    print("Task name",task)

    task_prompt = task_prompts[task]

    response_query = response if isinstance(response,str) else format_docs(response)

    # Agentic Task Chain
    agentic_task = (
        {
            "response": RunnablePassthrough(),
            "agents":  RunnableLambda(lambda _: ", ".join(agents))
        }
        | task_prompt
        | llm
        | StrOutputParser()
    )

    return agentic_task.invoke(response_query)



def coordinator(state: AgentState):
    """
    Coordinates the analysis process based on the given agent state.

    This function retrieves the appropriate analysis configuration for the given 
    analysis type in the agent state. It sets the custom query in the state if 
    the analysis type is "Custom Query"; otherwise, it uses the predefined query 
    from the configuration. The function then maps each agent in the configuration 
    to their respective task route, which includes "contract" for Contract Analyst, 
    "research" for Legal Researcher, and "strategy" for Legal Strategist.

    Args:
        state (AgentState): The current state of the agent, containing the analysis 
                            type, custom query, vectorstore, results, and reports.

    Returns:
        list: A list of task routes corresponding to the agents involved in the 
              analysis process.
    """

    config = analysis_configs.get(state["analysis_type"])
    if not config:
        raise ValueError(f"Invalid analysis type: {state['analysis_type']}")

    state["custom_query"] = state["custom_query"] if state["analysis_type"] == "Custom Query" else config["query"]

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
    """
    Executes the contract analysis task using the Contract Analyst agent.

    This function uses the agentic_rag function to perform the contract analysis 
    with the specified vectorstore and custom query extracted from the agent state.

    Args:
        state (AgentState): The current state of the agent, containing the vectorstore 
                            and custom query to be used for the analysis.

    Returns:
        dict: A dictionary containing the results of the contract analysis, 
              keyed by the agent's role ("Contract Analyst").
    """

    print("Agent Contract Analyst")
    return {
        "results": {
            "Contract Analyst": agentic_rag(
                vectorstore=state["vectorstore"], 
                task="contract", 
                custom_query=state["custom_query"]
            )
        }
    }

def run_research(state: AgentState):
    """
    Executes the legal research task using the Legal Researcher agent.

    This function leverages the agentic_rag function to conduct legal research
    based on the specified vectorstore and custom query extracted from the agent state.

    Args:
        state (AgentState): The current state of the agent, containing the vectorstore
                            and custom query to be used for the research.

    Returns:
        dict: A dictionary containing the results of the legal research, keyed by
              the agent's role ("Legal Researcher").
    """

    print("Agent Legal Researcher")
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
   
    """
    Executes the legal strategy task using the Legal Strategist agent.

    This function leverages the agentic_rag function to develop a legal strategy
    based on the specified vectorstore and custom query extracted from the agent state.

    Args:
        state (AgentState): The current state of the agent, containing the vectorstore
                            and custom query to be used for the strategy development.

    Returns:
        dict: A dictionary containing the results of the legal strategy development, keyed by
              the agent's role ("Legal Strategist").
    """
    
    print("Agent Legal Strategist")
    return {
        "results": {
            "Legal Strategist": agentic_rag(
                vectorstore=state["vectorstore"], 
                task="strategy", 
                custom_query=state["custom_query"])
        }
    }

def detail_analysis(state: AgentState):
    """
    Executes the detail analysis task using the appropriate agents.

    This function takes in the current state of the agent, containing the results of the
    previous analysis, and uses the agentic_task function to perform a detail analysis
    with the appropriate agents. The results are then stored in the state under "reports"
    with the key "details".

    Args:
        state (AgentState): The current state of the agent, containing the results of the
                            previous analysis.

    Returns:
        dict: A dictionary containing the results of the detail analysis, keyed by "reports"
              and then "details".
    """
    print("Agent Detail Analysis", sum(len(v) for v in state["results"].values()))
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
    """
    Executes the summary analysis task using the appropriate agents.

    This function takes in the current state of the agent, containing the results of the
    previous detail analysis, and uses the agentic_task function to perform a summary
    analysis with the appropriate agents. The results are then stored in the state under
    "reports" with the key "summary".

    Args:
        state (AgentState): The current state of the agent, containing the results of the
                            previous detail analysis.

    Returns:
        dict: A dictionary containing the results of the summary analysis, keyed by "reports"
              and then "summary".
    """
    print("Agent Summary Analysis", len(state["reports"]["details"]))
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
    """
    Executes the recommendation analysis task using the appropriate agents.

    This function takes in the current state of the agent, containing the results of the
    previous detail analysis, and uses the agentic_task function to perform a recommendation
    analysis with the appropriate agents. The results are then stored in the state under
    "reports" with the key "recommendation".

    Args:
        state (AgentState): The current state of the agent, containing the results of the
                            previous detail analysis.

    Returns:
        dict: A dictionary containing the results of the recommendation analysis, keyed by
              "reports" and then "recommendation".
    """
    print("Agent Recommendation Analysis", len(state["reports"]["details"]))
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

    """
    Merges the results from all parts of the analysis into one single dictionary.

    Args:
        state (AgentState): The current state of the agent, containing the results of the
                            previous analysis.

    Returns:
        dict: A dictionary containing the merged results of the analysis, keyed by "results".
    """

    merged = {}
    for part in state.get("results", {}).values():
        merged.update(part)
    return {"results": merged}

def build_langgraph():
    """
    Builds a StateGraph object representing the workflow of the legal analysis agent.

    This function constructs a StateGraph object that models the workflow of the legal
    analysis agent. The workflow consists of four nodes: contract, research, strategy, and
    detail. The contract node represents the contract analysis task, the research node
    represents the legal research task, the strategy node represents the legal strategy
    task, and the detail node represents the detail analysis task. The workflow also
    includes a summary node that represents the summary analysis task and a recommendation
    node that represents the recommendation analysis task.

    The workflow is constructed by adding nodes to the StateGraph object and then adding
    edges between the nodes. The edges are added in the following order:

    1. The contract node is connected to the detail node.
    2. The research node is connected to the detail node.
    3. The strategy node is connected to the detail node.
    4. The detail node is connected to the summary node.
    5. The detail node is connected to the recommendation node.
    6. The summary node is connected to the END node.
    7. The recommendation node is connected to the END node.

    The workflow is then compiled into a runnable object and returned.

    Returns:
        langchain_core.runnables.StateGraph: A StateGraph object representing the workflow
            of the legal analysis agent.
    """
    workflow = StateGraph(AgentState)
    
    workflow.add_node("contract", run_contract)
    workflow.add_node("research", run_research)
    workflow.add_node("strategy", run_strategy)
    workflow.add_node("detail", detail_analysis)
    workflow.add_node("summary", summary_analysis)
    workflow.add_node("recommendation", recommendation_analysis)

    workflow.add_conditional_edges(START, lambda state: coordinator(state), {
        "contract": "contract",
        "research": "research",
        "strategy": "strategy"
    })

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