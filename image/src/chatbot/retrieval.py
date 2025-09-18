import time
from typing import List, Dict, Any
import yaml

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

from .config import (
    PINECONE_API_KEY, RETRIEVAL_SUMMARY_CNT, EMBEDDING_DIMENSION
)
from .state import State

# Timing measurements for instrumentation
TIMINGS: Dict[str, float] = {}

# Load prompt templates
import os
prompt_template_path = os.path.join(os.path.dirname(__file__), "prompt_templates.yaml")
with open(prompt_template_path) as prompt_template_file:
    prompt_templates = yaml.safe_load(prompt_template_file)

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=EMBEDDING_DIMENSION)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("influencer-conversation-summary")
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

try:
    influencer_index = pc.Index("influencer-brain")
except Exception:
    influencer_index = None


def retrieve_context(state: State) -> State:
    """Retrieves relevant summaries of conversation from the vector store"""
    # Reset timings for each invocation and use raw last user message as the query if not provided
    TIMINGS.clear()
    t0 = time.time()
    query = state.get("user_query")
    if not query:
        chat_history = state.get("chat_history", [])
        query = chat_history[-1].content if chat_history else ""

    retrieved_docs = vector_store.similarity_search(
        query,
        k=RETRIEVAL_SUMMARY_CNT,
        filter={"user_id": state["user_id"]}
    )

    TIMINGS['retrieve_context'] = time.time() - t0

    retrieved_summaries = "\n\n".join([
        f"SUMMARY {i}:\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs, start=1)
    ])
    return {"retrieved_summaries": retrieved_summaries, "user_query": query}
