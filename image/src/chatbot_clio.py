from dotenv import load_dotenv

load_dotenv()

import yaml
import uuid
import re
import os


from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from supabase_utils import get_messages, create_message, get_total_messages_cnt_by_user, get_test_credentials


from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pprint import pprint


with open("prompt_templates.yaml") as prompt_template_file:
    prompt_templates = yaml.safe_load(prompt_template_file)

flags = re.DOTALL | re.MULTILINE
PATTERN_USER = re.compile(r"<user>(.*?)<\/user>", flags=flags)
PATTERN_ASSISTANT = re.compile(r"<assistant>(.*?)<\/assistant>", flags=flags)

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

RETRIEVAL_SUMMARY_CNT = 3
CONVERSATION_SUMMARY_THRESHOLD = 10
PAST_CHAT_HISTORY_CNT = 19
EMBEDDING_DIMENSION = 1024

llm = init_chat_model("gpt-4.1-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=EMBEDDING_DIMENSION)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("conversation-summaries")
vector_store = PineconeVectorStore(embedding=embeddings, index=index)


class State(TypedDict, total=False):
    """Defines the state of the chatbot conversation."""
    user_id: str
    chat_history: List[BaseMessage]
    msgs_cnt_by_user: int
    user_query: str
    retrieved_summaries: str
    response: str


def messages_to_txt(messages: List[BaseMessage]) -> str:
    role_map = {
        'human': 'user',
        'ai': 'assistant',
    }
    return "\n".join([
        f"{role_map[message.type].upper()}: {message.content}"
        for message in messages
    ])


def construct_user_query(state: State) -> State:
    prompt = prompt_templates['USER_QUERY_CONSTRUCTION_PROMPT']

    chat_history = state['chat_history']

    prompt = prompt.format(
        conversation=messages_to_txt(chat_history[:-1]),
        user_input=chat_history[-1].content
    )

    response = llm.invoke(prompt)

    return {'user_query': str(response.content)}


def retrieve_context(state: State) -> State:
    """Retrieves relevant summaries of conversation from the vector store"""
    retrieved_docs = vector_store.similarity_search(
        state["user_query"], 
        k=RETRIEVAL_SUMMARY_CNT,
        filter={"user_id": state["user_id"]}
    )
    
    retrieved_summaries = "\n\n".join([
        f"SUMMARY {i}:\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs, start=1)
    ])
    return {"retrieved_summaries": retrieved_summaries}


def generate_response(state: State) -> State:
    system_prompt = prompt_templates['MAIN_SYSTEM_PROMPT']
    system_prompt = system_prompt.format(summaries=state['retrieved_summaries'])
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state['chat_history']
    ])
    return {"response": str(response.content)}


def summarize(state: State) -> State:
    messages_cnt_by_user = state['msgs_cnt_by_user'] - 1

    if messages_cnt_by_user % CONVERSATION_SUMMARY_THRESHOLD == 0 and messages_cnt_by_user > 0:

        messages_to_summarize_txt = messages_to_txt(state['chat_history'][-(CONVERSATION_SUMMARY_THRESHOLD-1):-1])

        prompt = prompt_templates['SUMMARY_PROMPT']
        prompt = prompt.format(
            conversation=messages_to_summarize_txt
        )

        response = llm.invoke(prompt)
        
        metadata_payload = {
            "text": response.content,
            "user_id": state['user_id']
        }
        
        vectors = [(
            str(uuid.uuid4()), 
            embeddings.embed_query(response.content),
            metadata_payload
        )]
        index.upsert(vectors=vectors)
        
    return {}


graph_builder = StateGraph(State)

graph_builder.add_node(construct_user_query)
graph_builder.add_node(retrieve_context)
graph_builder.add_node(generate_response)
graph_builder.add_node(summarize)

graph_builder.add_edge(START, 'construct_user_query')
graph_builder.add_edge('construct_user_query', 'retrieve_context')
graph_builder.add_edge('retrieve_context', 'generate_response')
graph_builder.add_edge('generate_response', 'summarize')
graph_builder.add_edge('summarize', END)

chatbot_clio = graph_builder.compile()