#!/usr/bin/env python3
"""
Script to upload influencer data (reflections + memories) to Pinecone influencer-brain index.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
INDEX_NAME = "influencer-brain"

# Initialize
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def create_reflection_record(
    creator_id: str,
    role: str,  # "behav_econ", "psych", "political", "demo"
    theme: str,
    bullet: str,
    source_ids: List[str] = None,
    record_id: str = None
) -> Dict[str, Any]:
    """Create a reflection record for upload."""
    return {
        "id": record_id or f"ref-{uuid.uuid4()}",
        "type": "reflection",
        "creator_id": creator_id,
        "role": role,
        "theme": theme,
        "bullet": bullet,
        "source_ids": source_ids or [],
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

def create_memory_record(
    creator_id: str,
    text: str,
    source: str,
    platform: str = None,
    url: str = None,
    topics: List[str] = None,
    privacy_level: str = "public",
    record_id: str = None
) -> Dict[str, Any]:
    """Create a memory record for upload."""
    return {
        "id": record_id or f"mem-{uuid.uuid4()}",
        "type": "memory", 
        "creator_id": creator_id,
        "text": text,
        "source": source,
        "platform": platform,
        "url": url,
        "topics": topics or [],
        "privacy_level": privacy_level,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

def upload_records(records: List[Dict[str, Any]], batch_size: int = 100):
    """Upload records to Pinecone with embeddings."""
    print(f"Uploading {len(records)} records...")
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        vectors = []
        
        for record in batch:
            # Get text for embedding
            if record["type"] == "reflection":
                embed_text = record["bullet"]
            else:  # memory
                embed_text = record["text"]
            
            # Generate embedding
            embedding = embedding_model.encode(embed_text, normalize_embeddings=True).tolist()
            
            # Prepare metadata (exclude large fields)
            metadata = {k: v for k, v in record.items() if k != "id"}
            
            vectors.append({
                "id": record["id"],
                "values": embedding,
                "metadata": metadata
            })
        
        # Upload batch
        index.upsert(vectors=vectors)
        print(f"Uploaded batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size}")
    
    print("Upload complete!")

def load_from_json(file_path: str) -> List[Dict[str, Any]]:
    """Load records from JSON file."""
    if not Path(file_path).exists():
        print(f"File {file_path} not found")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_to_json(records: List[Dict[str, Any]], file_path: str):
    """Save records to JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    # Example: Create sample records for Taylor Swift
    sample_reflections = [
        create_reflection_record(
            creator_id="taylor_swift",
            role="psych",
            theme="Songwriting about personal growth",
            bullet="Frequently draws on personal milestones and relationships to explore themes of empowerment, reinvention, and creative control.",
            source_ids=["interview-rolling-stone-2023"]
        ),
        create_reflection_record(
            creator_id="taylor_swift", 
            role="behav_econ",
            theme="Business strategy",
            bullet="Prioritizes creative ownership and re-recording masters to maintain artistic and financial control.",
            source_ids=["business-insider-2021", "forbes-2022"]
        )
    ]
    
    sample_memories = [
        create_memory_record(
            creator_id="taylor_swift",
            text="Released the album 'Midnights' in October 2022 and launched the Eras Tour, reconnecting with longtime fans and introducing re-recordings of earlier albums.",
            source="public_post",
            platform="instagram",
            topics=["album_release", "tour", "re-recordings", "fan_engagement"],
            privacy_level="public"
        ),
        create_memory_record(
            creator_id="taylor_swift",
            text="I write songs in airports, on napkins, wherever inspiration strikes. The best songs come from the most unexpected moments.",
            source="interview",
            platform="podcast",
            topics=["songwriting", "creativity", "process"],
            privacy_level="public"
        )
    ]
    
    # Uncomment to upload sample data
    # all_records = sample_reflections + sample_memories
    # upload_records(all_records)
    
    # Save samples to JSON for reference
    save_to_json(sample_reflections, "sample_reflections.json")
    save_to_json(sample_memories, "sample_memories.json")
    print("Sample files created: sample_reflections.json, sample_memories.json")
