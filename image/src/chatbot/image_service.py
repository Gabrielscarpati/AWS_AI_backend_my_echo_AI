"""
Image processing service using GPT-4.1-mini for image description.
"""
import os
import base64
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from .models import llm

load_dotenv()


def describe_image(image_data: bytes) -> Optional[str]:
    """
    Describe an image using GPT-4.1-mini.

    Args:
        image_data: Base64 encoded image data

    Returns:
        Image description or None if failed
    """
    if not image_data:
        print("Empty image data provided")
        return None

    try:
        # Convert image data to base64 for the API call
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Create the message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please describe this image in detail. Focus on objects, people, setting, colors, and any text or important details that would help someone understand what's in the image. Keep the description factual and comprehensive but concise."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

        # Use the existing llm instance (gpt-4.1-mini)
        response = llm.invoke(messages)
        description = response.content.strip()

        if description:
            return description
        else:
            print("Empty image description result")
            return None

    except Exception as e:
        print(f"Error describing image: {e}")
        return None


def process_image_input(image_data: bytes) -> str:
    """
    Process image input and return description for database querying.

    Args:
        image_data: Base64 encoded image data

    Returns:
        Image description suitable for database querying
    """
    print("Processing image input...")

    description = describe_image(image_data)

    if description:
        print(f"Image described as: {description[:100]}...")
        return description
    else:
        print("Failed to describe image, using placeholder")
        return "An image was provided but could not be described"
