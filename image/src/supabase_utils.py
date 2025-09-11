from dotenv import load_dotenv
load_dotenv()

import os
import requests
import json

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

HEADERS = {
    'Content-Type': 'application/json',
    'User-Agent': 'insomnia/11.1.0',
    'apikey': SUPABASE_KEY
}

def create_message(
        content: str,
        is_user: bool,
        media_url: str = None,
        it_s_a_priority: bool = None,
        media_type: str = None,
        is_reflection: bool = None,
        personality_type: str = None,
        access_token: str = None
):
    if access_token is None:
        raise Exception("ACCESS TOKEN not provided")
    
    headers = HEADERS | {
        'Authorization': f"Bearer {access_token}"
    }
    
    url = f"{SUPABASE_URL}/rest/v1/rpc/create_message"
    payload = {
        "content": content,
        "is_user": is_user,
        "media_url": media_url,
        "it_s_a_priority": it_s_a_priority,
        "media_type": media_type,
        "is_reflection": is_reflection,
        "personality_type": personality_type,
    }
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()

def get_messages(limit: int = 10, offset: int = 0, access_token: str = None):
    if access_token is None:
        raise Exception("ACCESS TOKEN not provided")
    
    headers = HEADERS | {
        'Authorization': f"Bearer {access_token}"
    }
    
    url = f"{SUPABASE_URL}/rest/v1/rpc/get_message_paginated"
    payload = {
        "p_limit": str(limit),
        "p_offset": str(offset)
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def get_total_messages_cnt_by_user(access_token: str = None):
    if access_token is None:
        raise Exception("ACCESS TOKEN not provided")
    
    headers = HEADERS | {
        'Authorization': f"Bearer {access_token}"
    }
    
    url = f"{SUPABASE_URL}/rest/v1/rpc/get_total_messages_by_user"
    payload = {}
    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def get_test_credentials():
    """
    Returns (user_id, access_token)
    """
    response = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
        json={
            "email": "mateuslukas505@gmail.com",
            "password": "12345678"
        },
        headers=HEADERS
    )
    
    response.raise_for_status()
    response = response.json()
    
    user_id, access_token = response['user']['id'], response['access_token']
    return user_id, access_token


def get_curl_command():
    user_id, access_token = get_test_credentials()
    return (
        f'curl "http://localhost:9000/2015-03-31/functions/function/invocations" -d '
        f'\'{json.dumps({"user_id": user_id, "access_token": access_token})}\''
    )