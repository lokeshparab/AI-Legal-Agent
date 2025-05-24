from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as PC
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile, os, time
from pinecone import Pinecone, ServerlessSpec


def load_document_to_faiss(uploaded_file):

    # Initialize Pinecone
    

    pc = Pinecone()
    # Define your index name
    INDEX_NAME = "legal-doc-index"

    # Create index if it doesn't exist
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,  # Dimension for `jinaai/jina-embeddings-v2-base-en`
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

        # Load and split document
        docs = PyPDFLoader(path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        # Embed documents
        embedding = HuggingFaceEmbeddings(
            model_name="jinaai/jina-embeddings-v2-base-en", 
            model_kwargs={'trust_remote_code': True}
        )
        # Load into Pinecone
        vectorstore = PC.from_documents(
            documents=chunks,
            embedding=embedding,
            index_name=INDEX_NAME
        )

        return vectorstore
