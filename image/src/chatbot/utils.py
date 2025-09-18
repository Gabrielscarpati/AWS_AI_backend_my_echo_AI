from typing import List
from langchain_core.messages import BaseMessage


def messages_to_txt(messages: List[BaseMessage]) -> str:
    """Convert messages to text format."""
    role_map = {
        'human': 'user',
        'ai': 'assistant',
    }
    return "\n".join([
        f"{role_map[message.type].upper()}: {message.content}"
        for message in messages
    ])


def _truncate_lines(lines: List[str], max_chars: int) -> List[str]:
    """Truncate lines to fit within max_chars limit."""
    out, total = [], 0
    for ln in lines:
        if total + len(ln) > max_chars:
            break
        out.append(ln)
        total += len(ln)
    return out
