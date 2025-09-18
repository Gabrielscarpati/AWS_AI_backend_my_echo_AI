import uuid
from typing import List

from .state import State
from .config import CONVERSATION_SUMMARY_THRESHOLD
from .utils import messages_to_txt
from .retrieval import prompt_templates, embeddings, index
from .models import llm


def summarize(state: State) -> State:
    """Summarize conversation when threshold is reached."""
    # Summarize based on total message count (user + assistant) reaching multiples of threshold
    try:
        total_msgs_cnt = len(state.get('chat_history', []))
    except Exception:
        total_msgs_cnt = 0
    # Trigger when either the current total reaches a multiple of threshold, or
    # when adding the imminent assistant reply would reach it. This handles
    # different caller timings between local and prod flows.
    hits_now = (total_msgs_cnt % CONVERSATION_SUMMARY_THRESHOLD == 0)
    hits_with_next = ((total_msgs_cnt + 1) % CONVERSATION_SUMMARY_THRESHOLD == 0)
    is_summary_turn = total_msgs_cnt > 0 and (hits_now or hits_with_next)

    if not is_summary_turn:
        return {"summary_generated": False, "message_summary": ""}

    # Summarize the last N messages (by total messages)
    messages_to_summarize_txt = messages_to_txt(state.get('chat_history', [])[-CONVERSATION_SUMMARY_THRESHOLD:])

    prompt = prompt_templates['SUMMARY_PROMPT']
    prompt = prompt.format(
        conversation=messages_to_summarize_txt
    )

    response = llm.invoke(prompt)

    # Store in vector DB for retrieval
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

    return {"message_summary": str(response.content), "summary_generated": True}
