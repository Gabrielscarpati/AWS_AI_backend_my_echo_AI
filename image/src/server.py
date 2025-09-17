from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import json

# Load environment variables early
load_dotenv()

# Import the existing Lambda-style handler logic
from app import handler as lambda_handler

app = FastAPI(title="Mindhaven Chat API", version="1.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/version")
async def version():
    return {"version": "1.0.0", "description": "My latest changes"}


@app.post("/chat")
async def chat_endpoint(payload: dict):
    """
    Accepts the same JSON body as the Lambda handler's event.body.

    Example payload body:
    {
        "user_id": "...",
        "creator_id": "...",
        "influencer_name": "...",  # optional
        "influencer_personality_prompt": "...",  # optional
        "chat_history": [("user", "hi")],
        "msgs_cnt_by_user": 1
    }
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


