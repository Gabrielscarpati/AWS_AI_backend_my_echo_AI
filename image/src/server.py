from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import json
import os

# Load environment variables early
load_dotenv()

# Import the existing Lambda-style handler logic
from app import handler as lambda_handler

app = FastAPI(title="Mindhaven Chat API", version="1.0")

# Bearer token security scheme
security = HTTPBearer()

# Get the API token from environment variable
API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    raise RuntimeError("API_TOKEN environment variable is required")


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify the bearer token against the configured API token.
    Returns the token if valid, raises HTTPException if invalid.
    """
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/version")
async def version():
    return {"version": "1.0.6", "description": "Fixed chat history format conversion - now handles both string arrays and tuple arrays"}


@app.post("/chat")
async def chat_endpoint(payload: dict, token: str = Depends(verify_token)):
    """
    Accepts the same JSON body as the Lambda handler's event.body.
    Requires Bearer token authentication.
    
    Hardcoded for testing with mihir_ai:
    - creator_id: "mihir_ai"
    - influencer_name: "mihir"

    Example payload body:
    {
        "user_id": "...",
        "creator_id": "mihir_ai",  # Hardcoded for testing
        "influencer_name": "mihir",  # Hardcoded for testing
        "influencer_personality_prompt": "...",  # optional
        "chat_history": [("user", "hi")],
        "msgs_cnt_by_user": 1
    }
    
    Headers:
    Authorization: Bearer <your-api-token>
    """
    try:
        print(f"Received chat request with payload keys: {list(payload.keys())}")
        
        # Override with hardcoded values for testing
        payload["creator_id"] = "mihir_ai"
        payload["influencer_name"] = "mihir"
        
        # Set default personality prompt if not provided
        if "influencer_personality_prompt" not in payload or not payload["influencer_personality_prompt"]:
            payload["influencer_personality_prompt"] = (
                "Persona: You are Mihir. Speak with intelligence, technical expertise, and thoughtful analysis. "
                "Be helpful, precise, and supportive. Offer specific, practical guidance based on your experiences."
            )
        
        event = {
            "body": json.dumps(payload),
            "isBase64Encoded": False,
        }

        result = lambda_handler(event, {})
        status_code = result.get("statusCode", 500)
        body = result.get("body", "{}")
        
        try:
            data = json.loads(body)
        except Exception as e:
            print(f"Error parsing response body: {e}")
            data = {"raw_body": body}
            
        print(f"Returning response with status: {status_code}")
        return JSONResponse(content=data, status_code=status_code)
        
    except Exception as e:
        print(f"FastAPI endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Server error: {str(e)}"}, 
            status_code=500
        )


