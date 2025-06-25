from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Load docs from file (for demo, hardcoded path)
loader = PyPDFLoader("data/sample.pdf")  # You can later make this dynamic
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Embed & store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# Retrieval + generation
retriever = vectorstore.as_retriever()
llm = OpenAI()

def query_vector_store(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    if docs:
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"""
        return llm.predict(prompt)
    return None
