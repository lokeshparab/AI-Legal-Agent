
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def get_llm():
    return ChatGroq(groq_api_key="your-key", model="mixtral-8x7b-32768")

def make_chain(vectorstore, task: str):
    prompts = {
        "contract": "You are a contract analyst. {question}",
        "research": "You are a legal researcher. {question}",
        "strategy": "You are a legal strategist. {question}"
    }

    template = PromptTemplate.from_template(prompts[task])
    llm = get_llm()
    retriever = vectorstore.as_retriever()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": template}
    )

analysis_configs = {
    "Contract Review": {
        "query": "Review this contract and identify key terms, obligations, and potential issues.",
        "agents": ["Contract Analyst"],
        "description": "Detailed contract analysis focusing on terms and obligations"
    },
    "Legal Research": {
        "query": "Research relevant cases and precedents related to this document.",
        "agents": ["Legal Researcher"],
        "description": "Research on relevant legal cases and precedents"
    },
    "Risk Assessment": {
        "query": "Analyze potential legal risks and liabilities in this document.",
        "agents": ["Contract Analyst", "Legal Strategist"],
        "description": "Combined risk analysis and strategic assessment"
    },
    "Compliance Check": {
        "query": "Check this document for regulatory compliance issues.",
        "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
        "description": "Comprehensive compliance analysis"
    },
    "Custom Query": {
        "query": None,
        "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
        "description": "Custom analysis using all available agents"
    }
}
