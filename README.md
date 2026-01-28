# University Course Material RAG (AI Tutor)

An advanced RAG (Retrieval-Augmented Generation) platform for higher education. It transforms static PDF lecture slides into an interactive AI Tutor capable of explaining concepts, generating revision notes, and creating quizzes.

![Tech Stack](https://img.shields.io/badge/LlamaIndex-v0.10-blue) ![Tech Stack](https://img.shields.io/badge/Groq-Llama3-orange) ![Tech Stack](https://img.shields.io/badge/Streamlit-UI-red)

## Architecture
This project moves beyond standard "Chat with PDF" tutorials by implementing **Hierarchical Retrieval** and **Intent Routing**.

### Key Features
* **Hierarchical Indexing:** Splits data into "Parent" (Context) and "Child" (Search) chunks to ensure answers have full context.
* **Router Engine:** Dynamically selects the best tool based on user intent:
    * `Tutor`: Socratic explanations.
    * `Examiner`: Generates structured JSON for interactive quizzes.
    * `Summarizer`: Creates concise revision notes.
* **Multimodal Ingestion:** Uses **LlamaParse** to accurately read complex slide layouts (tables, diagrams) that standard OCR misses.

## Tech Stack
* **Orchestration:** LlamaIndex
* **LLM:** Groq llama-3.1-8b-instant
* **Vector DB:** ChromaDB (Persistent Local)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local/Free)
* **Frontend:** Streamlit

## Installation

1. Clone the repo
```bash
git clone [https://github.com/yourusername/edtech-rag.git](https://github.com/yourusername/edtech-rag.git)
cd edtech-rag
```
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Setup API Keys (.env)

```bash
GROQ_API_KEY=gsk_...
LLAMA_CLOUD_API_KEY=llx_...
``` 

4. Run the App

```bash
streamlit run app/main.py
```

![alt text](<Screenshot (29).png>)

![alt text](<Screenshot (28).png>)