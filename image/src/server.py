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
    return {"version": "1.0.1", "description": "My latest changes"}


@app.post("/chat")
async def chat_endpoint(payload: dict, token: str = Depends(verify_token)):
    """
    Accepts the same JSON body as the Lambda handler's event.body.
    Requires Bearer token authentication.

    Example payload body:
    {
        "user_id": "...",
        "creator_id": "...",
        "influencer_name": "...",  # optional
        "influencer_personality_prompt": "...",  # optional
        "chat_history": [("user", "hi")],
        "msgs_cnt_by_user": 1
    }
    
    Headers:
    Authorization: Bearer <your-api-token>
    """
    event = {
        "body": json.dumps(payload),
        "isBase64Encoded": False,
    }

    result = lambda_handler(event, {})
    status_code = result.get("statusCode", 500)
    body = result.get("body", "{}")
    try:
        data = json.loads(body)
    except Exception:
        data = {"raw_body": body}
    return JSONResponse(content=data, status_code=status_code)


