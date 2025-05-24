import streamlit as st, os
from packages.documents import load_document_to_faiss, load_document_to_pinecone
from packages.agents import build_langgraph
from packages.prompts import analysis_configs

import warnings
warnings.filterwarnings("ignore")

def init_session_state():
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = None
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    # if 'pinecone_api_key' not in st.session_state:
    #     st.session_state.pinecone_api_key = None

def main():
    init_session_state()
    st.title("📚 LangGraph Legal Agent Analyzer")


    with st.sidebar:
        st.header("🔑 API Configuration")
   
        st.session_state.groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state.groq_api_key if st.session_state.groq_api_key else "",
            help="Enter your OpenAI API key"
        )

        if st.session_state.groq_api_key:
            os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key

        hf_token = st.text_input(
            "Hugging Face API Key",
            type="password",
            value= "",
            help="Enter your Huggingface API key"
        )

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        
        # st.session_state.pinecone_api_key = st.text_input(
        #     "Pinecone API Key",
        #     type="password",
        #     value= "",
        #     help="Enter your Pinecone API key"
        # )

        # if st.session_state.pinecone_api_key:
        #     os.environ["PINECONE_API_KEY"] = st.session_state.pinecone_api_key

        
        st.divider()

        if st.session_state.groq_api_key:
            # Document upload section
            st.header("📄 Document Upload")
            uploaded_file = st.file_uploader("Upload Legal Document", type=['pdf'])

            if uploaded_file:
                with st.spinner("Processing document..."):
                    try:
                        st.session_state.vectorstore = load_document_to_faiss(uploaded_file)
                        # st.session_state.vectorstore = load_document_to_pinecone(uploaded_file)
                        
                    except Exception as e:
                            st.error(f"Error processing document: {str(e)}")

            st.divider()
            st.header("🔍 Analysis Options")
            analysis_type = st.selectbox(
                "Select Analysis Type",
                [
                    "Contract Review",
                    "Legal Research",
                    "Risk Assessment",
                    "Compliance Check",
                    "Custom Query"
                ]
            )
            st.session_state.legal_ai = build_langgraph()
        else:
            st.info("Enter Groq API key to upload documents")

            # Main content area
    if not st.session_state.vectorstore:
        st.info("👈 Waiting for connection...")
    elif not uploaded_file:
        st.info("👈 Please upload a legal document to begin analysis")
    elif st.session_state.legal_ai:
        # st.("Document Analysis")
        analysis_icons = {
            "Contract Review": "📑",
            "Legal Research": "🔍",
            "Risk Assessment": "⚠️",
            "Compliance Check": "✅",
            "Custom Query": "💭"
        }
        # Dynamic header with icon
        st.header(f"{analysis_icons[analysis_type]} {analysis_type} Analysis")
        st.header(f"📋 {analysis_configs[analysis_type]['description']}")
        st.write(f"🤖 Active Agents: {', '.join(analysis_configs[analysis_type]['agents'])}")

        if analysis_type == "Custom Query":
            custom_query = st.text_area(
                "Enter your specific query:",
                help="Add any specific questions or points you want to analyze"
            )
        else:
            custom_query = ""

        if st.button("Run Analysis"):
            
            
            final_state = st.session_state.legal_ai.invoke({
                "analysis_type": analysis_type,
                "custom_query": custom_query,
                "vectorstore": st.session_state.vectorstore,
                # "results": {},
                # "reports": {}
            })
            
            tabs = st.tabs(["Analysis", "Key Points", "Recommendations"])

            response = final_state["reports"]
            print(response)
            with tabs[0]:
                st.markdown("### Detailed Analysis")
                if "details" in response:
                    st.markdown(response['details'])
                else:
                    st.markdown("No detailed analysis available for this analysis type.")
            
            with tabs[1]:
                st.markdown("### Key Points")
                
                if "summary" in response:
                    st.markdown(response['summary'])
                else:
                    st.markdown("No detailed analysis available for this analysis type.")
            
            with tabs[2]:
                st.markdown("### Recommendations")
                
                if "recommendation" in response:
                    st.markdown(response['recommendation'])
                else:
                    st.markdown("No detailed analysis available for this analysis type.")
    else:
        st.info("Please upload a legal document to begin analysis")    

if __name__ == "__main__":
    
    main()