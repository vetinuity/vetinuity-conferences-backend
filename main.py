from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from datetime import datetime

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
    source_description: str
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

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Search
        results = await graphiti.search(request.query)
        
        # Build citations
        citations = []
        context = []
        
        for r in results[:5]:
            citations.append(Citation(
                source_description=r.fact if hasattr(r, 'fact') else str(r),
                conference="WVC",
                year=2025
            ))
            context.append(str(r))
        
        # Generate response
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a veterinary AI assistant. Answer based on conference proceedings."},
                {"role": "user", "content": f"Question: {request.query}\n\nContext: {' '.join(context)}\n\nAnswer:"}
            ]
        )
        
        return ChatResponse(
            response=response.choices[0].message.content,
            citations=citations,
            search_results_count=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
