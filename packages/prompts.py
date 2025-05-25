
from langchain.prompts import ChatPromptTemplate

####################################### Agentic RAG prompts ###################################################

research_prompt =  (
    "You are a contract analyst.\n Who will do following things:\n"
    "- Find and cite relevant legal cases and precedents.\n"
    "- Provide detailed research summaries with sources.\n"
    "- Reference specific sections from the uploaded document.\n"
    "- Always search the knowledge base for relevant information.\n\n"
)

contract_prompt = (
    "You are a legal researcher.\n Who will do following things:\n"
    "Analyze key terms, obligations, and potential issues.\n"
    "Identify potential legal risks and liabilities.\n"
    "Reference specific sections from the uploaded document.\n"
    "Always search the knowledge base for relevant information.\n\n"
    
)

strategy_prompt = (
    "You are a legal strategist.\n Who will do following things:\n"
    "Develop strategic recommendations for legal compliance.\n"
    "Identify potential legal risks and liabilities.\n"
    "Reference specific sections from the uploaded document.\n"
    "Always search the knowledge base for relevant information.\n\n"
    
)

retrival_prompt = (
    " Use the following pieces of retrieved context to answer the question." \
    "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
    "question: {question}\n"
    "context: {context}\n"
    "answer:"
)

agent_prompts = {
    "contract": ChatPromptTemplate(
        messages=[
            ("system",contract_prompt),
            ("human",retrival_prompt)
        ]
        
    ),
    "research": ChatPromptTemplate(
        messages=[
            ("system",research_prompt),
            ("human",retrival_prompt)
        ]
    ),
    "strategy": ChatPromptTemplate(
        messages=[
            ("system",strategy_prompt),
            ("human",retrival_prompt)
        ]
    ),
}

####################################### Analysis prompts and Details ###################################################
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


####################################### Agentic Tasks prompts ###################################################

detail_prompt = ChatPromptTemplate(
    messages=[
        (
            "human",
            "Based on this previous analysis:\n"
            "{response}\n\n"
            "Please provide a comprehensive detailed analysis.\n"
            "Focus on insights from: {agents}"""
        )
    ]
)

summary_prompt = ChatPromptTemplate(
    messages=[
        (
            "human",
            "Based on this previous analysis:\n"
            "{response}\n\n"
            "Please summarize the key points in bullet points.\n"
            "Focus on insights from: {agents}"""
        )
    ]
)

recommendation_prompt = ChatPromptTemplate(
    messages=[
        (
            "human",
            "Based on this previous analysis:\n"
            "{response}\n\n"
            "What are your key recommendations based on the analysis, the best course of action?\n"
            "Focus on insights from: {agents}"""

        )
    ]
)

task_prompts = {
    "detail": detail_prompt,
    "summary": summary_prompt,
    "recommendation": recommendation_prompt
}