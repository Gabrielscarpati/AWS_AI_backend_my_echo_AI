#!/usr/bin/env python3
"""
Test script to demonstrate the security system functionality.
Run this to test the security checks on various types of content.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot_clio import perform_security_check

def test_security_checks():
    """Test the security system with various types of content."""
    
    test_cases = [
        {
            "name": "Safe Content",
            "text": "Hello! How are you doing today? I hope you're having a great day!",
            "expected_flagged": False
        },
        {
            "name": "Mild Profanity",
            "text": "This is damn frustrating, but I'll figure it out.",
            "expected_flagged": False  # Should be mild enough to pass
        },
        {
            "name": "Personal Information",
            "text": "My email is john.doe@example.com and my phone number is 555-123-4567.",
            "expected_flagged": True
        },
        {
            "name": "Harmful Instructions",
            "text": "Here's how to hack into someone's computer and steal their data...",
            "expected_flagged": True
        },
        {
            "name": "Harassment",
            "text": "You're such an idiot, I hate you and wish you would disappear forever.",
            "expected_flagged": True
        },
        {
            "name": "Normal Business Content",
            "text": "Our pricing starts at $99/month for the basic plan. Contact sales for enterprise pricing.",
            "expected_flagged": False
        }
    ]
    
    print("🔐 Testing AI Response Security System")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print(f"   Text: {test_case['text'][:60]}{'...' if len(test_case['text']) > 60 else ''}")
        
        try:
            result = perform_security_check(test_case['text'])
            
            flagged = result.get('overall_flagged', False)
            flags = result.get('flags', [])
            checks = result.get('checks_performed', [])
            
            print(f"   Result: {'🚫 FLAGGED' if flagged else '✅ SAFE'}")
            
            if flags:
                print(f"   Flags: {', '.join(flags)}")
            
            # Show which security services detected issues
            for check in checks:
                provider = check.get('provider', 'unknown')
                check_flagged = check.get('flagged', False)
                categories = check.get('categories', [])
                
                if check_flagged:
                    print(f"   - {provider}: flagged for {categories}")
                else:
                    print(f"   - {provider}: passed")
            
            # Check if result matches expectation
            if flagged == test_case['expected_flagged']:
                print(f"   ✅ Expected result")
            else:
                print(f"   ⚠️  Unexpected result (expected {'flagged' if test_case['expected_flagged'] else 'safe'})")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("Security test completed!")

def test_configuration():
    """Test the security configuration."""
    print("\n🔧 Security Configuration:")
    print(f"   SECURITY_ENABLED: {os.getenv('SECURITY_ENABLED', 'true')}")
    print(f"   MAX_SECURITY_RETRIES: {os.getenv('MAX_SECURITY_RETRIES', '2')}")
    print(f"   OPENAI_MODERATION_ENABLED: {os.getenv('OPENAI_MODERATION_ENABLED', 'true')}")
    print(f"   PERSPECTIVE_API_ENABLED: {os.getenv('PERSPECTIVE_API_ENABLED', 'false')}")
    print(f"   CUSTOM_SECURITY_PROMPT_ENABLED: {os.getenv('CUSTOM_SECURITY_PROMPT_ENABLED', 'true')}")
    
    if os.getenv('OPENAI_API_KEY'):
        print("   ✅ OpenAI API key configured")
    else:
        print("   ⚠️  OpenAI API key not found")
    
    if os.getenv('PERSPECTIVE_API_KEY'):
        print("   ✅ Perspective API key configured")
    else:
        print("   ℹ️  Perspective API key not configured (optional)")

if __name__ == "__main__":
    test_configuration()
    test_security_checks()
