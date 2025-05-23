from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile, os

def load_document_to_faiss(uploaded_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        docs = PyPDFLoader(path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        embedding = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
        vectorstore = FAISS.from_documents(chunks, embedding)

        return vectorstore
