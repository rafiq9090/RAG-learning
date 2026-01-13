import os
import pdfplumber
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
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
    def __init__(
        self,
        pdf_path,
        persist_directory=None,
        chunk_size=None,
        chunk_overlap=None,
        retriever_k=None,
        retriever_score_threshold=None,
        model_name=None,
        min_context_chars=None,
        cite_sources=None,
        no_answer_response=None,
    ):
        if not pdf_path or not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError(f"Expected a PDF file, got: {pdf_path}")

        self.persist_directory = persist_directory or os.getenv("CHROMA_DIR", "./vector_db")
        self.chunk_size = int(chunk_size or os.getenv("CHUNK_SIZE", "1200"))
        self.chunk_overlap = int(chunk_overlap or os.getenv("CHUNK_OVERLAP", "200"))
        self.retriever_k = int(retriever_k or os.getenv("RETRIEVER_K", "6"))
        self.retriever_score_threshold = float(
            retriever_score_threshold or os.getenv("RETRIEVER_SCORE_THRESHOLD", "0.2")
        )
        self.min_context_chars = int(min_context_chars or os.getenv("MIN_CONTEXT_CHARS", "200"))
        self.cite_sources = self._parse_bool(
            cite_sources if cite_sources is not None else os.getenv("CITE_SOURCES"), True
        )
        self.no_answer_response = (
            no_answer_response
            or os.getenv("NO_ANSWER_RESPONSE", "I don't know based on the provided documents.")
        )

        # A. Setup Groq LLM (Llama 3.3 70B is excellent for RAG)
        self.llm = ChatGroq(
            model_name=model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,  # Keep at 0 for factual accuracy
            groq_api_key=os.getenv("GROQ_API_KEY"),
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
            "Include citations with page numbers when possible.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        retriever = self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.retriever_k,
                "score_threshold": self.retriever_score_threshold,
            },
        )
        self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    @staticmethod
    def _parse_bool(value, default=False):
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _process_pdf(self, pdf_path):
        if os.path.isdir(self.persist_directory) and os.listdir(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory, embedding_function=self.embeddings
            )

        all_docs = []
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 5,
            "snap_tolerance": 2,
        }
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                raise ValueError("The PDF has no pages.")
            for i, page in enumerate(pdf.pages):
                # Extract text and tables
                text = page.extract_text() or ""
                tables = page.extract_tables(table_settings=table_settings)

                if text.strip():
                    all_docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": pdf_path, "page": i + 1, "content_type": "text"},
                        )
                    )

                # Format tables as Markdown to help the LLM read the structure
                for table in tables:
                    if not table:
                        continue
                    rows = [
                        [str(cell) if cell is not None else "" for cell in row]
                        for row in table
                    ]
                    if not rows:
                        continue
                    header = rows[0]
                    separator = ["---"] * len(header)
                    body = rows[1:] if len(rows) > 1 else []
                    md_lines = [
                        "| " + " | ".join(header) + " |",
                        "| " + " | ".join(separator) + " |",
                    ]
                    for row in body:
                        md_lines.append("| " + " | ".join(row) + " |")
                    table_markdown = "\n".join(md_lines)
                    all_docs.append(
                        Document(
                            page_content=f"Table (page {i + 1}):\n{table_markdown}",
                            metadata={
                                "source": pdf_path,
                                "page": i + 1,
                                "content_type": "table",
                            },
                        )
                    )

        # Split into chunks that preserve table context
        text_docs = [doc for doc in all_docs if doc.metadata.get("content_type") == "text"]
        table_docs = [doc for doc in all_docs if doc.metadata.get("content_type") == "table"]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(text_docs) + table_docs

        return Chroma.from_documents(
            chunks, self.embeddings, persist_directory=self.persist_directory
        )

    def ask(self, query):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        # Modern invoke syntax
        response = self.rag_chain.invoke({"input": query})
        answer = response.get("answer", "").strip()
        docs = response.get("context", [])
        if not docs:
            return self.no_answer_response
        total_context_chars = sum(len(doc.page_content or "") for doc in docs)
        if total_context_chars < self.min_context_chars or not answer:
            return self.no_answer_response
        if not self.cite_sources:
            return answer
        citations = []
        for doc in docs:
            page = doc.metadata.get("page")
            source = os.path.basename(doc.metadata.get("source", ""))
            if page and source:
                citations.append(f"{source} p.{page}")
        if citations:
            citation_text = ", ".join(sorted(set(citations)))
            return f"{answer}\n\nSources: {citation_text}"
        return answer

# --- Run Test ---
if __name__ == "__main__":
    # Ensure research_paper.pdf is in the 'data' folder
    engine = RAGEngine("data/research_paper.pdf")

    # Test query from Table 1
    query = "According to Table 1, what is the Context Precision of BERT Chunking?"
    answer = engine.ask(query)
    print(f"\nAI Response: {answer}")
