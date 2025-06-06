from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings, JinaEmbeddings
import tempfile, os, time
from pinecone import Pinecone, ServerlessSpec

import warnings
warnings.filterwarnings("ignore")


def load_document_to_pinecone(uploaded_file):  
    """
    Load a PDF document into Pinecone.

    Args:
        uploaded_file (bytes): A PDF file.

    Returns:
        PineconeVectorStore: A vector store of the document.
    """
    
    # Initialize Pinecone
    pc = Pinecone()
    # Define your index name
    INDEX_NAME = "legal-doc-index"

    # Create index if it doesn't exist
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # Dimension for `all-MiniLM-L6-v2`
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    

    # Get Upload Documents
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        print("-"*80,"Load and Chunking","-"*80)
        # Load and split document
        docs = PyPDFLoader(path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        print("-"*80,"Load Embedding","-"*80)
        # Load Embedding
        embedding = JinaEmbeddings(
            jina_api_key=os.environ.get("JINA_API_KEY")
        )
        # Load into Pinecone
        index_name = pc.Index(INDEX_NAME)
        print("-"*80,"Ingesting Vector DB","-"*80)

        vectorstore = PineconeVectorStore(
            index=index_name,
            embedding=embedding
        )

        vectorstore.add_documents(chunks)



        return vectorstore


def load_document_to_faiss(uploaded_file):  
    """
    Load a PDF document into a FAISS vector store.

    Args:
        uploaded_file (bytes): A PDF file.

    Returns:
        FAISS: A vector store of the document.
    """

    # Get Upload Documents
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        print("-"*80,"Load and Chunking","-"*80)
        # Load and split document
        docs = PyPDFLoader(path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        print("-"*80,"Load Embedding","-"*80)
        # Load Embedding
        embedding = JinaEmbeddings(
            jina_api_key=os.environ.get("JINA_API_KEY")
        )
        # Load into FAISS
        print("-"*80,"Ingesting Vector DB","-"*80)
        vectorstore = FAISS.from_documents(chunks, embedding)


        return vectorstore
