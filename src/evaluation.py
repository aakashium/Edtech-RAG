import nest_asyncio
import pandas as pandas 
from raga.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Apply nest_asyncio
nest_asyncio.apply()

def generate_test_data():
    print("Generating synthetic test data for evaluation...")

    # Load Documents
    reader = SimpleDirectoryReader("./data", recursive=True)
    documents = reader.load_data()

    # Configure Generator
    genrator_llm = Groq(model="llama-3.1-8b-instant")
    critic_llm = Groq(model="llama-3.1-8b-instant")
    embeddings = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize Ragas Generator
    generator_llm = TestsetGenerator.from_llama_index(
        generator_llm = generator_llm,
        critic_llm = critic_llm,
        embeddings = embeddings
    )

    # Generate Questions
    testset = generator.generate_with_llama_index_docs(
        documents,
        test_size =5,
        distributions = {simple: 0.5, reasoning:0.3, multi_context: 0.2}
    )

    # Save to CSV
    df = testset.to_pandas()
    df.to_csv("gold_dataset", index=False)
    print("Genrated gold_dataset.csv with synthetic questions/anwser")
    return df

if __name__ == "__main__":
    generate_test_data()