# AI Prompt: Generate Influencer Data for Vector Database

You are tasked with creating comprehensive personality and memory data for influencers to populate a vector database. This data will be used to simulate authentic conversations.

## Data Types Needed:

### 1. REFLECTIONS (Personality insights)
- **Type**: "reflection" 
- **Role**: One of ["behav_econ", "psych", "political", "demo"]
  - `behav_econ`: Business decisions, pricing, sponsorships, deals
  - `psych`: Communication style, values, creative process, boundaries  
  - `political`: Political stances, controversial topics, red lines
  - `demo`: Audience demographics, engagement patterns, timing
- **Theme**: Brief category (e.g., "Creative process", "Brand partnerships")
- **Bullet**: Single insight sentence about their personality/approach
- **Source_ids**: Reference IDs (e.g., ["interview-vogue-2023"])

### 2. MEMORIES (Specific experiences)
- **Type**: "memory"
- **Text**: Specific quote, experience, or factual information
- **Source**: Where it came from ("interview", "public_post", "article")
- **Platform**: Social platform ("instagram", "tiktok", "podcast", "magazine")
- **Topics**: Relevant tags (["creativity", "business", "personal"])
- **Privacy_level**: "public" or "private"
- **URL**: Optional link

## Instructions:

1. **Use taylor swift as the influencer** 

2. **Create 15-20 REFLECTIONS** covering:
   - 4-5 behavioral economics insights (pricing, business decisions, partnerships)
   - 4-5 psychological traits (communication style, values, creative process)  
   - 3-4 political/controversial stances (what they avoid, support, boundaries)
   - 3-4 demographic insights (audience, timing, engagement patterns)

3. **Create 25-30 MEMORIES** including:
   - Direct quotes from interviews/posts
   - Specific career milestones  
   - Personal anecdotes they've shared
   - Business decisions they've made
   - Creative processes they've described
   - Audience interactions they've mentioned

4. **Output format**: JSON structure with two arrays:

```json
{
  "creator_id": "influencer_name",
  "reflections": [
    {
      "role": "psych",
      "theme": "Communication style", 
      "bullet": "Uses storytelling and personal vulnerability to connect with audience, often sharing behind-the-scenes moments.",
      "source_ids": ["interview-podcast-2023", "instagram-story-series"]
    }
  ],
  "memories": [
    {
      "text": "I remember sitting in my car after a bad meeting, crying, and posting about it. That post got 2M views because people related to the struggle.",
      "source": "interview",
      "platform": "podcast", 
      "topics": ["vulnerability", "business", "social_media"],
      "privacy_level": "public",
      "url": "https://podcast.example.com/episode-123"
    }
  ]
}
```

## Quality Guidelines:
- Make reflections **specific and actionable** - not generic personality traits
- Include **real quotes and experiences** where possible
- Ensure **diverse coverage** across all 4 role categories
- Make memories **concrete and detailed** - specific events, numbers, emotions
- Keep bullet points **under 150 characters**
- Keep memory text **under 300 characters**

## Example Influencer Suggestions:
- MrBeast (YouTube/Business)
- Emma Chamberlain (Lifestyle/Coffee)  
- Gary Vaynerchuk (Business/Marketing)
- Simone Biles (Sports/Mental Health)
- Ryan Kaji (Kid YouTuber/Family)
- Michelle Obama (Leadership/Advocacy)

Choose someone with rich public content and generate comprehensive data that captures their authentic voice and decision-making patterns.
