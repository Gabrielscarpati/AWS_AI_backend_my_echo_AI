#!/usr/bin/env python3
"""
Script to upload influencer data to Pinecone influencer-brain index.
Uses proper Pinecone structure with metadata nested under "metadata" key.
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


def load_json_data(file_path: str) -> Dict:
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def prepare_vectors(data: Dict, category: str, embedding_model) -> List[Dict]:
    """Prepare vectors for Pinecone indexing with proper structure."""
    vectors = []

    if category == "context_data":
        # Process context data items
        for i, context_item in enumerate(data.get("context_data", [])):
            text = context_item
            vector_id = f"context_{uuid.uuid4().hex}"
            embedding = embedding_model.encode(text).tolist()

            # Create vector with proper Pinecone structure
            vector = {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": text,
                    "category": "context_data",
                    "source": data.get("source", "unknown"),
                    "creator_id": data.get("creator_id", "unknown"),
                    "privacy_level": data.get("privacy_level", "public"),
                    "timestamp": datetime.now().isoformat()
                }
            }
            vectors.append(vector)

    elif category == "expert_analysis":
        # Process expert analysis items
        for analysis in data.get("expert_analysis", []):
            text = f"{analysis.get('title', '')}: {analysis.get('text', '')}"
            vector_id = f"analysis_{uuid.uuid4().hex}"
            embedding = embedding_model.encode(text).tolist()

            # Create vector with proper Pinecone structure
            vector = {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": text,
                    "category": "expert_analysis",
                    "title": analysis.get('title', ''),
                    "creator_id": data.get("creator_id", "unknown"),
                    "privacy_level": data.get("privacy_level", "public"),
                    "timestamp": datetime.now().isoformat()
                }
            }
            vectors.append(vector)

    elif category == "interview_and_communication_style":
        # Process interview and communication style data
        for style_data in data.get("interview_and_communication_style", []):
            # Process model scores
            model_type = style_data.get("model", "")
            scores = style_data.get("raw_scores", {})
            analysis_items = style_data.get("expert_analysis", [])

            # Create text from scores
            scores_text = f"{model_type} Scores: {json.dumps(scores)}"
            vector_id = f"style_{uuid.uuid4().hex}"
            embedding = embedding_model.encode(scores_text).tolist()

            # Create vector with proper Pinecone structure
            vector = {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": scores_text,
                    "category": "interview_and_communication_style",
                    "model_type": model_type,
                    "scores": json.dumps(scores),
                    "creator_id": data.get("creator_id", "unknown"),
                    "privacy_level": data.get("privacy_level", "public"),
                    "timestamp": datetime.now().isoformat()
                }
            }
            vectors.append(vector)

            # Process each analysis item — ensure each analysis entry is indexed with full metadata
            for analysis in analysis_items:
                # analysis may be a plain string or a dict with fields like {"text": ..., "title": ..., "source": ...}
                if isinstance(analysis, dict):
                    analysis_text_body = analysis.get("text", "")
                    analysis_title = analysis.get("title", "")
                    analysis_source = analysis.get("source", "")
                    analysis_additional = {k: v for k, v in analysis.items() if k not in ("text", "title", "source")}
                else:
                    analysis_text_body = str(analysis)
                    analysis_title = ""
                    analysis_source = ""
                    analysis_additional = {}

                analysis_text = f"{model_type} Analysis: {analysis_text_body}"
                analysis_id = f"analysis_{uuid.uuid4().hex}"
                analysis_embedding = embedding_model.encode(analysis_text).tolist()

                # Create vector with full metadata for each analysis item
                analysis_vector = {
                    "id": analysis_id,
                    "values": analysis_embedding,
                    "metadata": {
                        "text": analysis_text_body,
                        "full_text": analysis_text,
                        "category": "interview_and_communication_style",
                        "sub_category": "interview_analysis",
                        "model_type": model_type,
                        "title": analysis_title,
                        "source": analysis_source,
                        "additional": json.dumps(analysis_additional) if analysis_additional else "",
                        "creator_id": data.get("creator_id", "unknown"),
                        "privacy_level": data.get("privacy_level", "public"),
                        "timestamp": datetime.now().isoformat()
                    }
                }
                vectors.append(analysis_vector)

    return vectors


def main():
    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Initialize Pinecone
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    # Load and process data files
    data_files = [
        ("context_data.json", "context_data"),
        ("expert_analysis.json", "expert_analysis"),
        ("interview_and_communication_style.json", "interview_and_communication_style")
    ]

    all_vectors = []

    for file_name, category in data_files:
        if Path(file_name).exists():
            print(f"Processing {file_name}...")
            data = load_json_data(file_name)
            vectors = prepare_vectors(data, category, embedding_model)
            all_vectors.extend(vectors)
            print(f"  Added {len(vectors)} vectors from {file_name}")
        else:
            print(f"Warning: {file_name} not found")

    # Upload vectors to Pinecone in batches
    if all_vectors:
        print(f"Uploading {len(all_vectors)} vectors to Pinecone...")

        # Upload in batches of 100 (Pinecone's recommended batch size)
        batch_size = 100
        for i in range(0, len(all_vectors), batch_size):
            batch = all_vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"  Uploaded batch {i // batch_size + 1}/{(len(all_vectors) - 1) // batch_size + 1}")

        print("Upload completed successfully!")

        # Print summary
        categories = {}
        for vector in all_vectors:
            cat = vector["metadata"]["category"]
            categories[cat] = categories.get(cat, 0) + 1

        print("\nUpload summary:")
        for cat, count in categories.items():
            print(f"  {cat}: {count} vectors")
    else:
        print("No vectors to upload.")


if __name__ == "__main__":
    main()