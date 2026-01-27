import os
from dotenv import load_dotenv
import nest_asyncio

# LlamaIndex Core
from llama_index.core import PromptTemplate
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

from src.retrieval import load_index, get_llm
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Apply async
nest_asyncio.apply()
load_dotenv()

# Prompt Template
# 1. Tutor Persona 
TUTOR_PROMPT_STR = (
    "You are a helpful and patient University Tutor. \n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "If the student asks for an explanation, use analogies and simple language.\n"
    "Query: {query_str}\n"
    "Answer: "
)

# 2. Examiner Persona 
EXAM_PROMPT_STR = (
    "You are a strict Exam Generator. Your job is to create test questions based on the context.\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Generate a multiple-choice quiz based on the query instructions.\n"
    "YOU MUST RETURN ONLY VALID JSON. No conversational text.\n"
    "Format: \n"
    "[\n"
    "  {{'question': '...', 'options': ['a', 'b', 'c', 'd'], 'answer': '...', 'explanation': '...'}},\n"
    "]\n"
    "Query: {query_str}\n"
    "JSON Answer: "
)

# 3. Summarizer Persona 
SUMMARY_PROMPT_STR = (
    "You are a Revision Notes Generator. \n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Create detailed revision notes based on the query.\n"
    "Use distinct headers, bullet points, and bold key terms.\n"
    "Query: {query_str}\n"
    "Revision Notes: "
)

def get_router_engine():
    """
    Constructs the Router Engine that sits on top of 3 specialized Query Engines.
    """
    print("Initializing Router and Tools")

    # Setup Base Components
    llm = get_llm()
    index, storage_context = load_index() 

    # Base Retriever 
    base_retriever = index.as_retriever(similarity_top_k=6)
    retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=False)

    # Build Specialized Engines

    # Tutor Engine
    tutor_tmpl = PromptTemplate(TUTOR_PROMPT_STR)
    tutor_engine = RetrieverQueryEngine.from_args(
        retriever, llm=llm, text_qa_template= tutor_tmpl
    )

    # Exam Engine
    exam_tmpl = PromptTemplate(EXAM_PROMPT_STR)
    exam_engine = RetrieverQueryEngine.from_args(
        retriever, llm=llm, text_qa_template=exam_tmpl
    )

    # Summary Engine
    summary_tmpl = PromptTemplate(SUMMARY_PROMPT_STR)
    summary_engine = RetrieverQueryEngine.from_args(
        retriever, llm=llm, text_qa_template=summary_tmpl
    )

    # Define Tools
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_engine,
        description=(
            "Useful for creating revision notes, summaries or getting an overview of a topic."
            "Use thiss when the user asks to 'summarize', 'create notes', or 'review'."
        )
    )

    exam_tool = QueryEngineTool.from_defaults(
        query_engine=exam_engine,
        description=(
            "Useful for generating quiz questions, exams, or testing knowledge. "
            "Use this when the user asks for a 'quiz', 'test', 'exam', or 'questions'."
        ),
    )

    tutor_tool = QueryEngineTool.from_defaults(
        query_engine=tutor_engine,
        description=(
            "Useful for explaining concepts, answering specific questions, or general tutoring. "
            "Use this for everything else."
        ),
    )

    # Build the Router
    # LLMSingleSelector uses the LLM to choose the best single tool.
    router_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=[
            summary_tool,
            exam_tool,
            tutor_tool,
        ],
        verbose=True 
    )
    
    return router_engine

def test_routing():
    router = get_router_engine()
    
    queries = [
        "Explain the concept of entropy like I am 5.",
        "Generate 3 multiple choice questions about Thermodynamics.",
        "Create a revision sheet for the first lecture."
    ]
    
    for q in queries:
        print(f"\n────────────────────────────────────────")
        print(f"User: {q}")
        print(f"Router is thinking...")
        
        response = router.query(q)
        
        print(f"Response:\n{str(response)[:500]}...") 

if __name__ == "__main__":
    test_routing()