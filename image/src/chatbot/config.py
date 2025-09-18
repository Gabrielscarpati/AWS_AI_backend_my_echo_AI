import os
import re
from pathlib import Path

# Configuration constants
RETRIEVAL_SUMMARY_CNT = 3
CONVERSATION_SUMMARY_THRESHOLD = 9
PAST_CHAT_HISTORY_CNT = 9
EMBEDDING_DIMENSION = 1024

# Regex patterns
flags = re.DOTALL | re.MULTILINE
PATTERN_USER = re.compile(r"<user>(.*?)<\/user>", flags=flags)
PATTERN_ASSISTANT = re.compile(r"<assistant>(.*?)<\/assistant>", flags=flags)

# Environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
INDEX_DIR = Path(os.getenv("INFLUENCER_INDEX_DIR", "indexes"))
MEM_FILE = INDEX_DIR / "memories.json"
REF_FILE = INDEX_DIR / "reflections.json"

# Security configuration
SECURITY_ENABLED = os.getenv("SECURITY_ENABLED", "true").lower() in {"1", "true", "yes", "y"}
MAX_SECURITY_RETRIES = int(os.getenv("MAX_SECURITY_RETRIES", "2"))
OPENAI_MODERATION_ENABLED = os.getenv("OPENAI_MODERATION_ENABLED", "true").lower() in {"1", "true", "yes", "y"}
PERSPECTIVE_API_ENABLED = os.getenv("PERSPECTIVE_API_ENABLED", "false").lower() in {"1", "true", "yes", "y"}
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")
CUSTOM_SECURITY_PROMPT_ENABLED = os.getenv("CUSTOM_SECURITY_PROMPT_ENABLED", "true").lower() in {"1", "true", "yes", "y"}

# Lens keywords for influencer retrieval
LENS_KEYWORDS = {
    "behav_econ": ["price", "budget", "cpm", "roi", "usage", "rights", "whitelist", "format", "deliverable", "contract",
                   "payment", "sponsorship", "accept", "deal"],
    "psych": ["tone", "voice", "style", "boundaries", "values", "ethics", "creative", "script", "tone of voice", "dm",
              "intro"],
    "political": ["politics", "controversial", "avoid", "red line", "endorsement", "mlm", "diet"],
    "demo": ["audience", "demographic", "age", "country", "region", "when", "time", "timezone", "peak", "engagement"],
}
ORDER = ["behav_econ", "psych", "demo", "political"]
