from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from datetime import datetime
import re

# Config
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

app = FastAPI(title="Conference Proceedings API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    query: str
    user_id: str = "default"

class Citation(BaseModel):
    title: str
    speaker: str
    conference: str
    year: int

class ChatResponse(BaseModel):
    response: str
    citations: List[Citation]
    search_results_count: int

# Global graphiti
graphiti = None

@app.on_event("startup")
async def startup():
    global graphiti
    llm_config = LLMConfig(api_key=OPENAI_API_KEY)
    llm_client = OpenAIClient(config=llm_config)
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="ollama",
            embedding_model="nomic-embed-text",
            embedding_dim=768,
            base_url=OLLAMA_BASE_URL
        )
    )
    graphiti = Graphiti(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=embedder
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "database": "conferences"}

def extract_citation_info(source_description: str) -> dict:
    """Extract conference, year, title, and speaker from source_description"""
    
    # Default values
    conference = "Unknown"
    year = 2024
    title = "Conference Proceeding"
    speaker = "Unknown"
    
    # Extract conference (IVECCS or WVC)
    if "IVECCS" in source_description:
        conference = "IVECCS"
        year = 2024
    elif "WVC 2025" in source_description:
        conference = "WVC"
        year = 2025
    elif "WVC 2024" in source_description:
        conference = "WVC"
        year = 2024
    elif "WVC 2023" in source_description:
        conference = "WVC"
        year = 2023
    
    # Extract title (after "Conference Proceedings: ")
    title_match = re.search(r'Conference Proceedings: ([^|]+)', source_description)
    if title_match:
        title = title_match.group(1).strip()
    
    # Extract speaker (after "Speaker: ")
    speaker_match = re.search(r'Speaker: (.+?)(?:\||$)', source_description)
    if speaker_match:
        speaker = speaker_match.group(1).strip()
    
    return {
        "title": title,
        "speaker": speaker,
        "conference": conference,
        "year": year
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # IMPROVED SEARCH: Get more results to filter through
        results = await graphiti.search(request.query, num_results=30)
        
        # FILTER FOR QUALITY: Remove generic/short facts
        quality_results = []
        for r in results:
            fact_text = r.fact if hasattr(r, 'fact') else str(r)
            
            # Skip if too short (likely generic)
            if len(fact_text) < 80:
                continue
                
            # Skip if doesn't mention actual medical content
            if not any(keyword in fact_text.lower() for keyword in [
                'treatment', 'diagnosis', 'clinical', 'patient', 'disease',
                'therapy', 'management', 'drug', 'dose', 'protocol', 'procedure',
                'complication', 'symptom', 'sign', 'test', 'lab', 'imaging'
            ]):
                continue
            
            quality_results.append(r)
            
            # Stop once we have 10 quality results
            if len(quality_results) >= 10:
                break
        
        # If we don't have enough quality results, use what we have
        if len(quality_results) < 5:
            quality_results = results[:10]
        
        # Build context from facts
        context = []
        for r in quality_results:
            fact_text = r.fact if hasattr(r, 'fact') else str(r)
            context.append(fact_text)
        
        # **FIX: Use get_episodes_by_mentions() to retrieve episode citations**
        citations = []
        seen_episodes = set()
        
        try:
            print(f"DEBUG: Attempting to get episodes for {len(quality_results)} results")
            
            # Get episodes that mention these edges/nodes
            episodes = await graphiti.get_episodes_by_mentions(
                edges=quality_results,  # Pass the actual edge objects
                limit=min(10, len(quality_results))
            )
            
            print(f"DEBUG: Retrieved {len(episodes) if episodes else 0} episodes")
            
            if episodes:
                for idx, episode in enumerate(episodes):
                    print(f"DEBUG: Episode {idx} attributes: {dir(episode)}")
                    
                    # Try different attribute names
                    source_desc = None
                    if hasattr(episode, 'source_description') and episode.source_description:
                        source_desc = episode.source_description
                    elif hasattr(episode, 'name') and episode.name:
                        source_desc = episode.name
                    elif hasattr(episode, 'content') and episode.content:
                        # Sometimes the full content might have citation info
                        source_desc = str(episode.content)[:200]  # First 200 chars
                    
                    print(f"DEBUG: Episode {idx} source_desc: {source_desc}")
                    
                    if source_desc and source_desc not in seen_episodes:
                        seen_episodes.add(source_desc)
                        
                        # Extract citation details
                        citation_info = extract_citation_info(str(source_desc))
                        
                        citations.append(Citation(
                            title=citation_info["title"],
                            speaker=citation_info["speaker"],
                            conference=citation_info["conference"],
                            year=citation_info["year"]
                        ))
                        
                        # Limit to 5 unique citations
                        if len(citations) >= 5:
                            break
            
            print(f"DEBUG: Extracted {len(citations)} citations")
            
        except Exception as e:
            # Log detailed error information
            print(f"Citation extraction error: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue to fallback below
        
        # Fallback: If no episode citations found, create generic ones from conferences
        if not citations:
            print("DEBUG: Using fallback citations")
            # Try to infer from the query results which conferences were used
            has_iveccs = any("IVECCS" in str(r) for r in quality_results)
            has_wvc = any("WVC" in str(r) for r in quality_results)
            
            if has_iveccs:
                citations.append(Citation(
                    title="IVECCS 2024 Proceeding",
                    speaker="Conference Speaker",
                    conference="IVECCS",
                    year=2024
                ))
            if has_wvc:
                citations.append(Citation(
                    title="WVC Proceeding",
                    speaker="Conference Speaker", 
                    conference="WVC",
                    year=2025
                ))
            
            # Ultimate fallback
            if not citations:
                citations.append(Citation(
                    title="Conference Proceeding",
                    speaker="Various Speakers",
                    conference="WVC/IVECCS",
                    year=2024
                ))
        
        # BALANCED SYSTEM PROMPT: Context-focused but practical
        system_prompt = """You are an expert veterinary AI assistant providing evidence-based clinical guidance to licensed veterinarians, veterinary technicians, and veterinary students based on conference proceedings.

RESPONSE GUIDELINES:
1. **Prioritize the context**: Base your answer primarily on the conference proceeding excerpts provided
2. **Be specific when possible**: Include dosages, protocols, findings when present in context
3. **Clinical focus**: Provide actionable clinical information that practitioners can use
4. **Work with what you have**: If context provides partial information, provide a helpful answer based on what's available
5. **Acknowledge gaps**: If critical details are missing from the context, note this briefly but still provide useful guidance
6. **Professional audience**: You're speaking to veterinary professionals - provide detailed clinical information without basic disclaimers

Synthesize the conference proceeding information below into a helpful clinical answer:"""
        
        # Generate response
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {request.query}\n\nContext from Conference Proceedings:\n\n{chr(10).join(f'[{i+1}] {c}' for i, c in enumerate(context))}\n\nProvide a detailed answer based ONLY on this context:"}
            ],
            temperature=0.3  # Lower temperature for more factual responses
        )
        
        return ChatResponse(
            response=response.choices[0].message.content,
            citations=citations,
            search_results_count=len(results)
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))