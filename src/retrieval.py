import os
from dotenv import load_dotenv
import nest_asyncio

# LlamaIndex Core
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank

# Integrations
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import chromadb

# Apply nest_asyncio
nest_asyncio.apply()

# Load env 
load_dotenv()

# Configuration
PERSIST_DIR = "./storage"
CHROMA_DB_PATH = "./chroma_db"
COLLENTION_NAME = "edtech_rag"

def get_llm():
    """
    Initialize Groq LLM.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in .env file")

    return Groq(model="llama3-70b-8192", temperature=0.1)

def load_index():
    """
    Recontructs the index from ChromaDB and DocStore.
    """
    print("Loading index from storage")

    # Initialize Embedding Model
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Re-connect to ChromaDB
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLENTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Load the DocStore
    storage_context = StorageContext.from_defaults(
        persist_dir=PERSIST_DIR,
        vector_store=vector_store
    )

    # Load the Index
    index = load_index_from_storage(
        storage_context,
        embed_model=embed_model
    )
    return index, storage_context

def get_query_engine():
    """
    Builds the AutoMerging Engine.
    """
    # Load Index and Storage Context
    index, storage_context = load_index()
    llm = get_llm()

    # Retriever Strategy
    # Base Vector Retriever - Finds top 10 most similar Leaf Nodes
    base_retriever = index.as_retriever(similarity_top_k=6)

    # Auto Merging Retriever - takes the base retriever's results. 
    retriever = AutoMergingRetriever(
        base_retriever,
        storage_context,
        verbose=True
    ) 

    # Reranker - Reorders the retrieved nodes by relevance to ensure the best context hits the LLM first.abs
    reranker = SentenceTransformerRerank(top_n=3, model="BAAI/bge-reranker-base")

    # Construct the Query Engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        llm=llm,
        node_postprocessors=[reranker]
    )

    return query_engine

def test_retrival(question):
    print(f"\n Asking: {question}")
    engine = get_query_engine()

    # Execute Query
    response = engine.query(question)

    # Output Response
    print("\n Answer:\n")
    print(response)

    # Sources 
    print("\n Sources Used:")
    for node in response.source_nodes:
        # Check if we got the Parent or Child
        content_len = len(node.node.get_content())
        print(f"- Node ID: {node.node.node_id[:8]}... | Length: {content_len} chars | Score: {node.score:.4f}")

if __name__ == "__main__":
    test_retrival("Explain the main concepts covered in the slide.")


