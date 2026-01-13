# RAG Engine with Table Support

This is a Retrieval-Augmented Generation (RAG) system built with **LangChain** and **Groq** (using Llama 3). It is designed to extract information from PDF documents, with a specific focus on accurately parsing and indexing **tables** for better retrieval performance.

## Features

- **Advanced PDF Parsing**: Uses `pdfplumber` to extract text and formatted tables.
- **Table Support**: converts tables within PDFs into Markdown format before indexing, allowing the LLM to understand row/column relationships better.
- **High-Performance LLM**: Integreate with Groq API for ultra-fast inference using `llama-3.3-70b-versatile`.
- **Local Embeddings**: Uses `HuggingFaceEmbeddings` (all-MiniLM-L6-v2) for cost-effective and local vector generation.
- **Vector Database**: Uses **ChromaDB** for efficient similarity search and retrieval.

## Prerequisites

- Python 3.8+
- A [Groq Cloud](https://console.groq.com/) API Key.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rafiq9090/RAG-learning.git
   cd RAG_project
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the root directory:
   ```bash
   touch .env
   ```

2. Add your Groq API key:
   ```env
   GROQ_API_KEY=your_gsk_key_here
   ```

3. Optional tuning parameters:
   ```env
   # Model used by Groq
   GROQ_MODEL=llama-3.3-70b-versatile

   # Vector DB persistence
   CHROMA_DIR=./vector_db

   # Chunking controls
   CHUNK_SIZE=1200
   CHUNK_OVERLAP=200

   # Retrieval controls
   RETRIEVER_K=6
   RETRIEVER_SCORE_THRESHOLD=0.2

   # Answer safety
   MIN_CONTEXT_CHARS=200
   CITE_SOURCES=true
   NO_ANSWER_RESPONSE=I don't know based on the provided documents.
   ```

## Usage

1. **Prepare your Data:**
   Place your PDF file (e.g., `research_paper.pdf`) inside the `data/` directory.

2. **Run the Engine:**
   ```bash
   python rag_engine.py
   ```

3. **Modify Queries:**
   Open `rag_engine.py` and scroll to the bottom to change the test query:
   ```python
   if __name__ == "__main__":
       engine = RAGEngine("data/research_paper.pdf")
       query = "Your question here..." 
       print(engine.ask(query))
   ```

4. **Citations:**
   Answers include page citations when available:
   ```text
   AI Response: ...
   
   Sources: research_paper.pdf p.3, research_paper.pdf p.5
   ```

## Sample Demo

Here is an example of the engine extracting precise data from a table in the research paper:

**Input Query:**
> "According to Table 1, what is the Context Precision of BERT Chunking?"

**System Output:**
```text
AI Response: According to Table 1, the Context Precision of BERT Chunking is 92%.
```
**Input Query:**
> "What is the best chunking method for accuracy?"

**System Output:**
```text
AI Response: According to the provided context, the best chunking method for accuracy is BERT Chunking, with a precision of 92%, recall of 85%, and relevancy of 89% [8]. It also has the highest answer faithfulness score of 94% among the three methods compared. 

Here is the comparison table for reference:
Method | Precision | Recall | Relevancy | Answer Faithfulness
------|----------|--------|------------|-------------------
BERT Chunking | 92%    | 85%    | 89%       | 94%
Recursive Chunking | 85% | 78%    | 82%       | 88%
Token Chunking | 76% | 82%    | 79%       | 81% 

Please note that the accuracy of the chunking method may depend on the specific use case and system requirements. [8]

Sources: research_paper.pdf p.2, research_paper.pdf p.3
```

**Input Query:**
> "Which method has the highest faithfulness score?"

**System Output:**
```text
AI Response: The highest faithfulness score is 94%, achieved by BERT Chunking.

Sources: research_paper.pdf p.3
```

## Structure

- `rag_engine.py`: Main logic for PDF loading, embedding, and querying.
- `vector_db/`: Directory where Chroma stores the indexed data.
- `data/`: Directory for input PDF files.
