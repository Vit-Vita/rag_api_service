# main.py
import uuid
from fastapi import FastAPI, HTTPException, Body, Depends, Path, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session

import rag_logic
import database as db
import config

# Initialize the FastAPI app
app = FastAPI(
    title="Multi-Company RAG API",
    description="An API for a multi-company RAG system using PostgreSQL and PGVector.",
    version="3.1.0"
)

# --- API Key Security Setup ---
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    """Checks if the provided API key is valid."""
    if api_key == config.settings.API_SECRET_KEY:
        return api_key
    else:
        raise HTTPException(
            status_code=401, 
            detail="Invalid or missing API Key"
        )

# --- Pydantic Models ---
class GeneralChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class SpecificChatRequest(BaseModel):
    question: str
    job_id: int
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str

class Job(BaseModel):
    JobID: int
    Title: str
    Location: Optional[str] = None
    Workload: Optional[str] = None

    class Config:
        from_attributes = True # Updated from orm_mode

class JobListResponse(BaseModel):
    jobs: List[Job]

# --- API Endpoints ---
@app.get("/{company_name}/jobs", response_model=JobListResponse, tags=["Jobs"])
def get_all_jobs(
    company_name: str = Path(..., description="Company name, e.g., 'Hager' or 'Luzerner Kantonalbank'"),
    db_session: Session = Depends(db.get_db),
    api_key: str = Depends(get_api_key)
):
    """Retrieves a list of all available job postings for a specific company."""
    jobs_from_db = rag_logic.load_all_jobs(db_session, company_name)
    if not jobs_from_db:
        raise HTTPException(status_code=404, detail=f"No jobs found for company: {company_name}")
    
    return JobListResponse(jobs=[Job.from_orm(job) for job in jobs_from_db])

# --- General Chat Endpoint ---
@app.post("/{company_name}/chat/general", response_model=ChatResponse, tags=["Chat"])
def process_general_chat(
    request: GeneralChatRequest,
    company_name: str = Path(..., description="Company name, e.g., 'Hager' or 'LUKB'"),
    db_session: Session = Depends(db.get_db),
    api_key: str = Depends(get_api_key)
):
    """Handles general questions not related to a specific job posting."""
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        answer = rag_logic.generate_response(
            db_session=db_session,
            user_input=request.question,
            session_id=session_id,
            company_name=company_name,
            job_id=None
        )
        return ChatResponse(answer=answer, session_id=session_id)
    except Exception as e:
        print(f"An error occurred during general chat processing: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

# --- Specific Chat Endpoint ---
@app.post("/{company_name}/chat/specific", response_model=ChatResponse, tags=["Chat"])
def process_specific_chat(
    request: SpecificChatRequest,
    company_name: str = Path(..., description="Company name, e.g., 'Hager' or 'LUKB'"),
    db_session: Session = Depends(db.get_db),
    api_key: str = Depends(get_api_key)
):
    """Handles questions that are focused on a specific job posting."""
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        answer = rag_logic.generate_response(
            db_session=db_session,
            user_input=request.question,
            session_id=session_id,
            company_name=company_name,
            job_id=request.job_id
        )
        return ChatResponse(answer=answer, session_id=session_id)
    except Exception as e:
        print(f"An error occurred during specific chat processing: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")