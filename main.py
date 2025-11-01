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
        
        # Build citations from episodes (not just facts)
        citations = []
        context = []
        seen_episodes = set()
        
        for r in quality_results:
            # Get the fact text
            fact_text = r.fact if hasattr(r, 'fact') else str(r)
            context.append(fact_text)
            
            # Try to get episode information
            if hasattr(r, 'episodes') and r.episodes:
                for episode in r.episodes:
                    if hasattr(episode, 'source_description') and episode.source_description:
                        episode_id = episode.source_description
                        
                        # Avoid duplicate citations
                        if episode_id not in seen_episodes:
                            seen_episodes.add(episode_id)
                            
                            # Extract citation details
                            citation_info = extract_citation_info(episode.source_description)
                            
                            citations.append(Citation(
                                title=citation_info["title"],
                                speaker=citation_info["speaker"],
                                conference=citation_info["conference"],
                                year=citation_info["year"]
                            ))
                            
                            # Limit to 5 unique citations
                            if len(citations) >= 5:
                                break
                
                if len(citations) >= 5:
                    break
        
        # Fallback: If no episode citations found, create generic ones
        if not citations:
            citations = [Citation(
                title="Conference Proceeding",
                speaker="Various Speakers",
                conference="WVC/IVECCS",
                year=2024
            )]
        
        # IMPROVED SYSTEM PROMPT: Stricter about using only context
        system_prompt = """You are an expert veterinary AI assistant providing evidence-based clinical guidance to licensed veterinarians, veterinary technicians, and veterinary students based on conference proceedings.

CRITICAL RULES:
1. **ONLY use information from the provided context** - Do NOT use your general veterinary knowledge
2. **If the context doesn't contain the answer**, respond with: "The available conference proceedings do not contain specific information about this topic. This may not have been covered in WVC 2023-2025 or IVECCS 2024."
3. **Be specific** - Include exact dosages, protocols, findings when present in context
4. **No generic advice** - Don't provide textbook veterinary information not in the context

RESPONSE GUIDELINES (when context IS available):
1. **Clinical Specificity**: Provide specific information from the proceedings:
   - Drug dosages with calculations if mentioned
   - Specific lab findings if discussed
   - Imaging characteristics if described
   - Procedural steps if outlined

2. **Differential Diagnoses**: Only if discussed in the proceedings

3. **Step-by-Step Protocols**: Only if protocols are provided in context

4. **Clinical Context**: Your audience consists of veterinary professionals - do NOT include generic disclaimers like "consult your veterinarian"

5. **Cite the source**: When possible, reference which speaker or session the information came from

Answer the following question using ONLY the provided conference proceeding context. If the context is insufficient, say so explicitly."""
        
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
        raise HTTPException(status_code=500, detail=str(e))
