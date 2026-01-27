import os 
import nest_asyncio
from dotenv import load_dotenv

# LlamaIndex core
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import Settings

# Integration
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_parse import LlamaParse
import chromadb

# Async execution
nest_asyncio.apply()

# Load API keys
load_dotenv()

# Configuration
DATA_DIR ="./data"
PERSIST_DIR = "./storage"
CHROMA_DB_PATH = "./chroma_db"
COLLENTION_NAME = 'edtech_rag'

def pipeline_components():
    """
    Initialize the embedding model and node parser.
    """
    print("Initializing pipeline components")

    # Embedding Model 
    # Using all-MiniLM-L6-v2 
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Hierarchical Node Parser
    # Creates a tree
    # Root (2048 tokens): Holds the full concept context
    # Leaf (512 tokens): Small chunks help precise vector search
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128]
    )

    # Set global settings
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    return embed_model, node_parser


def load_and_parse_documents():
    """
    Uses LlamaParse to understand complex PDF slides.
    Falls back to standard loader for text files.
    """

    # Initialize LlamaParse
    parser = LlamaParse(
        result_type = "markdown",
        verbose = True,
        language="en"
    )

    # File Extractor maps .pdf to LlamaParse
    file_extractor = {".pdf":parser}

    # Load data
    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        file_extractor=file_extractor,
        recursive=True
    )
    documents = reader.load_data()
    print(f"Loaded {len(documents)} document pages.")
    
    return documents

def run_ingestion():
    # Setup
    embed_model, node_parser = pipeline_components()

    # Load Documents (OCR & Parsing)
    documents =load_and_parse_documents()

    # Create Nodes (Hierarchical Splitting)
    print("Splitting documents into hierarchical nodes")
    nodes = node_parser.get_nodes_from_documents(documents)

    leaf_nodes = get_leaf_nodes(nodes)
    print(f"Created {len(nodes)} total nodes (Hierarchy) -> {len(leaf_nodes)} leaf nodes to index.")

    # Initialize Chromadb Vector Store
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLENTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Initialize Docstore 
    docstore = SimpleDocumentStore()

    # Add all nodes (parents + children) to DocStore
    docstore.add_documents(nodes)

    # Create Storage Context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore
    )

    # Indexing 
    print("Embedding and indexing leaf nodes")
    index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    # Persist to disk 
    storage_context.persist(persist_dir=PERSIST_DIR)

    print("Ingestion Complete... Data ready for quering")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)

    run_ingestion()