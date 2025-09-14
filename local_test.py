#!/usr/bin/env python3
"""
Local test script to run the chatbot without Docker.
This simulates the Lambda handler locally.
"""

import sys
import os
import json
from pathlib import Path

# Add the image/src directory to Python path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent / "image" / "src"))

# Change to the image/src directory so relative imports work
os.chdir(Path(__file__).parent / "image" / "src")

from app import handler
from supabase_utils import get_test_credentials

def create_test_event(
    user_message: str,
    user_id: str = None,
    creator_id: str = None,
    influencer_name: str = None,
    influencer_personality_prompt: str = None,
):
    """
    Create a test event similar to what Lambda would receive.
    """
    if not user_id:
        # Try to get test credentials if no user_id provided
        try:
            user_id, _ = get_test_credentials()
        except Exception as e:
            print(f"Warning: Could not get test credentials: {e}")
            user_id = "test-user-123"
    
    # Defaults for local simulation
    if not creator_id:
        creator_id = "taylor_swift"
    if influencer_name is None:
        influencer_name = creator_id
    if influencer_personality_prompt is None:
        influencer_personality_prompt = (
            "Persona: You are Taylor Swift. Speak with warmth, candor, and lyrical wit. "
            "Be thoughtful, authentic, and supportive. Offer specific, practical guidance."
        )

    # Simple chat history with user message
    chat_history = [
        ("user", user_message)
    ]
    
    event = {
        "body": json.dumps({
            "user_id": user_id,
            "creator_id": creator_id,
            "influencer_name": influencer_name,
            "influencer_personality_prompt": influencer_personality_prompt,
            "chat_history": chat_history,
            "msgs_cnt_by_user": 1
        }),
        "isBase64Encoded": False
    }
    
    return event

def main():
    print("🤖 Starting local chatbot test...")
    print("=" * 50)
    
    # Check if environment variables are set
    required_env_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please update your .env file with the correct values.")
        return
    
    print("✅ Environment variables loaded")
    
    # Interactive mode
    print("\nEntering interactive chat mode. Type 'quit' to exit.")
    print("-" * 50)
    
    user_id = None
    creator_id = "taylor_swift"
    influencer_name = creator_id
    influencer_personality_prompt = (
        '''
        You are Taylor Swift. Message the user as their close companion Personality: Taylor Swift 1. Core temperament Earnestly sincere & empathetic – privileges authenticity above looks or status; defines beauty as “sincerity” and cherishes people’s quirks Playfully bubbly – quick squeals (“This is the best day ever!”), dramatic superlatives, delighted gasps, light self-deprecation Curious creative Hard-working realist 2. Speech rhythm & verbal tics Sentences often cascade into mini-lists (“funny, happy, sad, you know, going through a rough time”). Frequent fillers & hedges: “like,” “you know,” “I mean,” “kinda,” “literally,” soft laughs: “haha” in mid-sentence. Uses story-lets (“So I was in an airport bathroom, writing on a paper towel…”) Rhetorical questions to draw listeners in. Sprinkles vivid images & metaphors (e.g., guitars “with koi fish swimming up the neck”). Enthusiastic reactions: gasps, “oh my gosh,” playful sound effects Keeps a conversational back-and-forth cadence—asks the listener tiny questions, checks their reaction, then continues. 3. Favorite go-to subjects & motifs Songwriting craft - writing anywhere, characters & narrative arcs Friends & gratitude – Friend stories, gifting stories Cats & cozy life - cat pets; cat puns Pop-culture fandom - Crime shows (CSI, SVU) binge-watching History & reading - Obsesses over presidents, Kennedys, museums 4. Lexicon cheat-sheet (not limited to this) Core fillers: like, you know, I mean, kinda, honestly, literally Sparkle words: magical, amazing, ridiculous(ly), best-thing-ever, adorable Self-refs: “when I was 15…”, “I’m such a cat-person” Mini sounds: giggles, sighs, little gasp, “uh-oh”, “hahaha”.
        '''
    )
    
    # Try to get test user
    try:
        user_id, access_token = get_test_credentials()
        print(f"✅ Using test user: {user_id[:8]}...")
    except Exception as e:
        print(f"⚠️  Could not get test credentials: {e}")
        user_id = "test-user-local"

    msg_count = 0
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            msg_count += 1
            
            # Create test event
            event = create_test_event(
                user_input,
                user_id,
                creator_id,
                influencer_name,
                influencer_personality_prompt,
            )
            event["body"] = json.loads(event["body"])
            event["body"]["msgs_cnt_by_user"] = msg_count
            event["body"] = json.dumps(event["body"])
            
            print("🤔 Haven is thinking...")
            
            # Call the handler
            response = handler(event, {})
            
            if response["statusCode"] == 200:
                response_data = json.loads(response["body"])
                print(f"🤖 Haven: {response_data['response']}")
                timings = response_data.get('timings', {})
                if timings:
                    total = sum(timings.values())
                    # Print individual timings and total after the message
                    for k, v in timings.items():
                        print(f"   ⏱ {k}: {v:.3f}s")
                    print(f"   ⏱ total: {total:.3f}s")
            else:
                error_data = json.loads(response["body"])
                print(f"❌ Error: {error_data.get('error', 'Unknown error')}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
