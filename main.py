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
    source_description: str

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
    elif "WVC) 2025" in source_description or "WVC 2025" in source_description:
        conference = "WVC"
        year = 2025
    elif "WVC) 2024" in source_description or "WVC 2024" in source_description:
        conference = "WVC"
        year = 2024
    elif "WVC) 2023" in source_description or "WVC 2023" in source_description:
        conference = "WVC"
        year = 2023
    
    # Extract title - IVECCS format: "IVECCS 2024 Conference Proceedings: TITLE | Speaker:"
    iveccs_title_match = re.search(r'IVECCS 2024 Conference Proceedings: ([^|]+)', source_description)
    if iveccs_title_match:
        title = iveccs_title_match.group(1).strip()
    else:
        # WVC format: look for quoted title in middle
        wvc_title_match = re.search(r'"([^"]+)"', source_description)
        if wvc_title_match:
            title = wvc_title_match.group(1).strip()
    
    # Extract speaker - after "Speaker:" or look for names with credentials
    speaker_match = re.search(r'Speaker: ([^|]+?)(?:\n|$)', source_description, re.DOTALL)
    if speaker_match:
        speaker = speaker_match.group(1).strip()
    else:
        # Try to find name with DVM credentials
        name_match = re.search(r'([A-Z][a-z]+ (?:[A-Z]\. )?[A-Z][a-z]+(?:,? (?:DVM|Dr[A-Z][a-z]+|[A-Z]{3,}))+)', source_description)
        if name_match:
            speaker = name_match.group(1).strip()
    
    return {
        "title": title,
        "speaker": speaker,
        "conference": conference,
        "year": year
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Search for relevant edges
        results = await graphiti.search(request.query, num_results=30)
        
        # Filter for quality
        quality_results = []
        for r in results:
            fact_text = r.fact if hasattr(r, 'fact') else str(r)
            
            if len(fact_text) < 80:
                continue
                
            if not any(keyword in fact_text.lower() for keyword in [
                'treatment', 'diagnosis', 'clinical', 'patient', 'disease',
                'therapy', 'management', 'drug', 'dose', 'protocol', 'procedure',
                'complication', 'symptom', 'sign', 'test', 'lab', 'imaging'
            ]):
                continue
            
            quality_results.append(r)
            
            if len(quality_results) >= 10:
                break
        
        if len(quality_results) < 5:
            quality_results = results[:10]
        
        # Build context
        context = []
        for r in quality_results:
            fact_text = r.fact if hasattr(r, 'fact') else str(r)
            context.append(fact_text)
        
        # WORKING FIX: Query Neo4j directly for citations
        from neo4j import GraphDatabase
        
        citations = []
        seen_citations = set()
        
        try:
            # Extract edge UUIDs
            edge_uuids = [r.uuid for r in quality_results if hasattr(r, 'uuid')]
            
            if edge_uuids:
                # Direct Neo4j query to get source_description from episodes
                driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
                
                with driver.session() as session:
                    # Query to find episodes that mention these edges
                    result = session.run("""
                        MATCH (ep:Episodic)-[:MENTIONS]->(e1:Entity)-[r:RELATES_TO]->(e2:Entity)
                        WHERE r.uuid IN $edge_uuids
                        RETURN DISTINCT ep.source_description as citation
                        LIMIT 10
                    """, edge_uuids=edge_uuids)
                    
                    for record in result:
                        citation_text = record["citation"]
                        if citation_text and citation_text not in seen_citations:
                            seen_citations.add(citation_text)
                            
                            citation_info = extract_citation_info(citation_text)
                            
                            citations.append(Citation(
                                title=citation_info["title"],
                                speaker=citation_info["speaker"],
                                conference=citation_info["conference"],
                                year=citation_info["year"]
                                source_description=citation_text
                            ))
                            
                            if len(citations) >= 5:
                                break
                
                driver.close()
                
        except Exception as e:
            print(f"Citation extraction error: {e}")
        
        # Fallback
        if not citations:
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
            
            if not citations:
                citations.append(Citation(
                    title="Conference Proceeding",
                    speaker="Various Speakers",
                    conference="WVC/IVECCS",
                    year=2024
                ))
        
        # Generate response
        system_prompt = """You are an expert veterinary AI assistant providing evidence-based clinical guidance to licensed veterinarians, veterinary technicians, and veterinary students based on conference proceedings.

RESPONSE GUIDELINES:
1. **Prioritize the context**: Base your answer primarily on the conference proceeding excerpts provided
2. **Be specific when possible**: Include dosages, protocols, findings when present in context
3. **Clinical focus**: Provide actionable clinical information that practitioners can use
4. **Work with what you have**: If context provides partial information, provide a helpful answer based on what's available
5. **Acknowledge gaps**: If critical details are missing from the context, note this briefly but still provide useful guidance
6. **Professional audience**: You're speaking to veterinary professionals - provide detailed clinical information without basic disclaimers

Synthesize the conference proceeding information below into a helpful clinical answer:"""
        
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {request.query}\n\nContext from Conference Proceedings:\n\n{chr(10).join(f'[{i+1}] {c}' for i, c in enumerate(context))}\n\nProvide a detailed answer based ONLY on this context:"}
            ],
            temperature=0.3
        )
        
        return ChatResponse(
            response=response.choices[0].message.content,
            citations=citations,
            search_results_count=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))