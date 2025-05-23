from langchain_groq import ChatGroq
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
