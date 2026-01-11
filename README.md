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

## Sample Demo

Here is an example of the engine extracting precise data from a table in the research paper:

**Input Query:**
> "According to Table 1, what is the Context Precision of BERT Chunking?"

**System Output:**
```text
AI Response: According to the table, the Context Precision of BERT Chunking is 92%.
```
**Input Query:**
> "What is the best chunking method for accuracy?"

**System Output:**
```text
AI Response: According to the provided context, BERT Chunking has the highest accuracy, with a precision of 92%, recall of 85%, relevancy of 89%, and faithfulness of 94%. Therefore, BERT Chunking appears to be the best chunking method for accuracy among the three methods compared (Recursive Chunking, BERT Chunking, and Token Chunking).
```

## Structure

- `rag_engine.py`: Main logic for PDF loading, embedding, and querying.
- `vector_db/`: Directory where Chroma stores the indexed data.
- `data/`: Directory for input PDF files.
