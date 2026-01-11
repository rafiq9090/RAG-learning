import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA

# Load API Key from .env
load_dotenv()
# Ensure GROQ API key is present for Groq model usage
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY not set. Add it to .env (GROQ_API_KEY=your_key_here) or export it in your shell before running.")

class RAGEngine:
    def __init__(self, pdf_path):
        # 1. Setup Groq LLM
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile", # High performance model
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # 2. Setup Embeddings (Runs locally on your CPU for free)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 3. Process the Document
        self.vector_db = self._process_pdf(pdf_path)
        
        # 4. Create the QA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever()
        )

    def _process_pdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Chunking: split into 1000 character pieces
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        # Save to local vector store
        return Chroma.from_documents(chunks, self.embeddings, persist_directory="./vector_db")

    def ask(self, query):
        return self.qa_chain.invoke(query)

# --- Quick Test ---
if __name__ == "__main__":
    engine = RAGEngine("data/research_paper.pdf")
    response = engine.ask("What is the main summary of this document?")
    print(f"Groq Response: {response['result']}")