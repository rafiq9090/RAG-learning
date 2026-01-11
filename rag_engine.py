import os
import pdfplumber
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# 1. Load Environment Variables
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY not set. Please add it to your .env file.")

class RAGEngine:
    def __init__(self, pdf_path):
        # A. Setup Groq LLM (Llama 3.3 70B is excellent for RAG)
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0, # Keep at 0 for factual accuracy
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # B. Setup Embeddings (Runs locally)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # C. Process the Document with Table Support
        self.vector_db = self._process_pdf(pdf_path)
        
        # D. Setup Modern Retrieval Chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(self.vector_db.as_retriever(), question_answer_chain)

    def _process_pdf(self, pdf_path):
        all_docs = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract text and tables
                text = page.extract_text() or ""
                tables = page.extract_tables()
                
                # Format tables as Markdown to help the LLM read the structure
                table_markdown = ""
                for table in tables:
                    for row in table:
                        clean_row = [str(cell) if cell else "" for cell in row]
                        table_markdown += "| " + " | ".join(clean_row) + " |\n"
                
                content = f"{text}\n\n{table_markdown}"
                all_docs.append(Document(page_content=content, metadata={"source": pdf_path, "page": i+1}))
        
        # Split into chunks that preserve table context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks = text_splitter.split_documents(all_docs)
        
        return Chroma.from_documents(chunks, self.embeddings, persist_directory="./vector_db")

    def ask(self, query):
        # Modern invoke syntax
        response = self.rag_chain.invoke({"input": query})
        return response["answer"]

# --- Run Test ---
if __name__ == "__main__":
    # Ensure research_paper.pdf is in the 'data' folder
    engine = RAGEngine("data/research_paper.pdf")
    
    # Test query from Table 1
    query = "Compare the Answer Relevancy of Token Chunking versus Recursive Chunking as listed in the paper."
    answer = engine.ask(query)
    print(f"\nAI Response: {answer}")