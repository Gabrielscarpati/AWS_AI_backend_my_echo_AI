# AI Response Security Implementation

## Overview

This implementation adds a comprehensive security layer to check AI-generated messages for malicious content before they're sent to users. The system uses multiple security APIs and includes automatic retry mechanisms when content is flagged.

## Features

### Multi-Layer Security Checks

1. **OpenAI Moderation API** (Default: Enabled)
   - Checks for harassment, hate speech, sexual content, violence, etc.
   - Fast and reliable, built into OpenAI's ecosystem
   - Categories: harassment, hate, self-harm, sexual, violence, etc.

2. **Google Perspective API** (Default: Disabled - requires API key)
   - Advanced toxicity detection
   - Granular scoring for different types of harmful content
   - Categories: toxicity, severe toxicity, identity attack, insult, profanity, threat, etc.

3. **Custom LLM Security Prompt** (Default: Enabled)
   - Uses your existing LLM to analyze content
   - Looks for personal information leaks, manipulation attempts, misinformation
   - Customizable and context-aware

### Automatic Retry System

- When content is flagged, the system automatically regenerates the response with enhanced safety instructions
- Configurable retry limit (default: 2 attempts)
- If max retries are reached, returns a safe fallback message
- Lower temperature generation for safety-focused responses

### Workflow Integration

The security system is integrated into the LangGraph workflow as follows:

```
generate_influencer_answer → security_check_node → [conditional]
                                     ↓
                          ┌─────────────────────────┐
                          ↓                         ↓
                regenerate_safe_response    →    summarize
                          ↓
                security_check_node (retry)
```

## Configuration

### Environment Variables

```env
# Enable/disable security checks (default: true)
SECURITY_ENABLED=true

# Maximum number of retry attempts (default: 2)
MAX_SECURITY_RETRIES=2

# OpenAI Moderation API (default: true)
OPENAI_MODERATION_ENABLED=true

# Google Perspective API (requires API key)
PERSPECTIVE_API_ENABLED=false
PERSPECTIVE_API_KEY=your_perspective_api_key_here

# Custom LLM security prompt (default: true)
CUSTOM_SECURITY_PROMPT_ENABLED=true
```

### Getting API Keys

#### Google Perspective API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Perspective Comment Analyzer API
4. Create credentials (API key)
5. Add the key to your environment variables

## Response Format

The API response now includes security information:

```json
{
  "response": "AI generated response",
  "summary_generated": false,
  "message_summary": "",
  "security": {
    "security_check_passed": true,
    "security_flags": [],
    "security_retry_count": 0,
    "security_check_result": {
      "overall_flagged": false,
      "flags": [],
      "checks_performed": [
        {
          "flagged": false,
          "categories": [],
          "scores": {...},
          "provider": "openai_moderation"
        }
      ],
      "security_enabled": true,
      "check_duration": 0.234
    }
  },
  "timings": {...},
  "timings_total": 1.234,
  "wall_time": 1.456
}
```

## Security Check Details

### OpenAI Moderation Categories
- `harassment`: Content that expresses, incites, or promotes harassing language
- `hate`: Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste
- `self-harm`: Content that promotes, encourages, or depicts acts of self-harm
- `sexual`: Content meant to arouse sexual excitement or promote sexual services
- `violence`: Content that depicts death, violence, or physical injury

### Perspective API Categories
- `toxicity`: General toxicity score
- `severe_toxicity`: Severe toxicity that could be very harmful
- `identity_attack`: Attacks based on identity characteristics
- `insult`: Insulting, inflammatory, or negative comments
- `profanity`: Swear words, curse words, or other obscene language
- `threat`: Threats of violence or harm

### Custom LLM Security Checks
- Personal information leaks (emails, phone numbers, addresses, SSN)
- Malicious instructions or manipulation attempts
- Inappropriate sexual content
- Hate speech or discrimination
- Violence or threats
- Misinformation or false claims
- Attempts to bypass safety measures
- Requests for illegal activities
- Phishing or scam attempts
- Content that could harm minors

## Usage Examples

### Basic Usage
The security system works automatically once enabled. No code changes are needed for basic functionality.

### Monitoring Security Events
```python
# Check if a response was flagged
response = api_call(...)
security_info = response['security']

if not security_info['security_check_passed']:
    print(f"Content was flagged for: {security_info['security_flags']}")
    print(f"Retries attempted: {security_info['security_retry_count']}")
```

### Custom Security Thresholds
You can modify the thresholds in the `check_perspective_api` function:

```python
thresholds = {
    'TOXICITY': 0.7,        # Lower = more strict
    'SEVERE_TOXICITY': 0.5,
    'IDENTITY_ATTACK': 0.6,
    # ... etc
}
```

## Performance Considerations

- **OpenAI Moderation**: ~100-200ms per check
- **Perspective API**: ~200-500ms per check  
- **Custom LLM Prompt**: ~1-3s per check (depends on model)

Security checks run in parallel where possible to minimize latency impact.

## Troubleshooting

### Common Issues

1. **API Key Not Working**
   - Verify your API keys are correct
   - Check API quotas and billing
   - Ensure APIs are enabled in respective consoles

2. **High False Positive Rate**
   - Adjust thresholds in the security functions
   - Disable overly sensitive checks
   - Review custom security prompt for over-strictness

3. **Performance Issues**
   - Disable slower security checks (Perspective API, Custom LLM)
   - Increase timeout values
   - Consider caching for repeated content

### Debugging

Enable detailed logging by adding print statements in security functions or checking the `security_check_result` in the API response.

## Security Best Practices

1. **Defense in Depth**: Use multiple security layers
2. **Regular Updates**: Keep security prompts and thresholds updated
3. **Monitoring**: Track security flags and false positives
4. **Graceful Degradation**: Always provide fallback responses
5. **User Experience**: Balance security with usability

## Future Enhancements

- Rate limiting for repeated security violations
- User-specific security profiles
- Machine learning-based custom classifiers
- Integration with external security services
- Detailed security analytics and reporting
