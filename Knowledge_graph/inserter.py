import os
import json
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm import ollama_model_complete
from lightrag.llm import ollama_embedding
from lightrag.llm import openai_complete
from lightrag.llm import openai_complete_if_cache
from dotenv import load_dotenv

load_dotenv()


def insert_custom_kg_safely(rag, custom_kg):
    """
    Safely insert a custom knowledge graph into LightRAG by handling the event loop appropriately.

    Args:
        rag: LightRAG instance
        custom_kg: Dictionary containing the custom knowledge graph data
    """
    try:
        # Check if we're in IPython/Jupyter
        try:
            import IPython
            ipy = IPython.get_ipython()
            if ipy is not None:
                import nest_asyncio
                nest_asyncio.apply()
        except ImportError:
            pass

        # Create and run the event loop if one isn't already running
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the insertion
        return loop.run_until_complete(rag.ainsert_custom_kg(custom_kg))

    except Exception as e:
        print(f"Error inserting custom KG: {str(e)}")
        raise

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("base_url"),
        **kwargs
    )

# Initialize LightRAG
def initialize_lightrag(working_dir='./test'):
    return LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embedding(
                texts,
                embed_model="nomic-embed-text"
            ),
        ),
    )


# Usage example
if __name__ == "__main__":
    # Load custom KG data
    with open('data/cutomKG.json', 'r') as f:
        custom_kg_data = json.load(f)

    # Initialize RAG
    rag = initialize_lightrag()

    # Insert custom KG
    insert_custom_kg_safely(rag, custom_kg_data)
