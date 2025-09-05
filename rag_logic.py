import datetime
from typing import List, Optional
from sqlalchemy.orm import Session

from langchain_community.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import database as db
from config import settings

# --- 1. Initialize Global Components ---
# These are loaded once and reused across requests.
embeddings = OpenAIEmbeddings(model=settings.OPENAI_EMBED_MODEL)
llm = ChatOpenAI(model=settings.OPENAI_CHAT_MODEL, temperature=0)

# --- 2. Database Functions ---
def load_all_jobs(db_session: Session, company_name: str) -> List[db.Job]:
    """Loads all job records for a specific company from the relational database."""
    company = db_session.query(db.Company).filter(db.Company.Name.ilike(f"%{company_name}%")).first()
    if not company:
        return []
    return company.jobs

def save_conversation(db_session: Session, session_id: str, question: str, answer: str, company_name: str, job_id: Optional[int] = None):
    """Saves the user question and AI answer to the database."""
    # Find the company
    company = db_session.query(db.Company).filter(db.Company.Name.ilike(f"%{company_name}%")).first()
    if not company:
        # Or create it if it doesn't exist, though it should from ingestion
        company = db.Company(Name=company_name)
        db_session.add(company)
        db_session.flush()

    # Find or create a chat session
    chat = db_session.query(db.Chat).filter_by(SessionID=session_id).first()
    if not chat:
        chat = db.Chat(
            JobID=job_id,
            SessionID=session_id,
            CreatedAt=datetime.datetime.now()
        )
        db_session.add(chat)
        db_session.flush()

    # Save user message
    user_message = db.Message(ChatID=chat.ChatID, SenderType="user", Content=question, Time=datetime.datetime.now())
    # Save AI message
    ai_message = db.Message(ChatID=chat.ChatID, SenderType="ai", Content=answer, Time=datetime.datetime.now())
    
    db_session.add_all([user_message, ai_message])
    db_session.commit()

# --- 3. RAG Logic ---
prompt_template = """
You are a helpful assistant answering questions from a potential applicant for a job at {company_name}.
Use the following context to answer the question. If you don't know the answer, say so.
Answer in the same language as the question.

Context:
{context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)

def get_combined_context(question: str, company_name: str, job_id: Optional[int]) -> str:
    """Builds context by retrieving from the correct PGVector collections."""
    company_slug = company_name.lower()
    
    # Connection to PGVector
    vectorstore = PGVector(
        connection_string=settings.DATABASE_URL,
        embedding_function=embeddings,
        collection_name=f"{company_slug}_job_descriptions" # Start with jobs
    )
    
    # 1. Retrieve job-specific documents
    job_docs = []
    if job_id:
        # More precise retrieval using metadata filter
        job_retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"job_id": str(job_id)}})
        job_docs = job_retriever.get_relevant_documents(question)
    else:
        # General job search
        job_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        job_docs = job_retriever.get_relevant_documents(question)
        
    # 2. Retrieve general company documents from the other collection
    vectorstore.collection_name = f"{company_slug}_general_info"
    general_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    general_docs = general_retriever.get_relevant_documents(question)
    
    # 3. Format context string
    parts = []
    if job_docs:
        parts.append("Relevant Job Postings:\n" + "\n\n".join(d.page_content for d in job_docs))
    if general_docs:
        parts.append("General Company Information:\n" + "\n\n".join(d.page_content for d in general_docs))
        
    return "\n\n---\n\n".join(parts) if parts else "No relevant context found."

# --- 4. Main Response Generation ---
def generate_response(db_session: Session, user_input: str, session_id: str, company_name: str, job_id: Optional[int]) -> str:
    """Generates a response and saves the conversation."""
    context = get_combined_context(user_input, company_name, job_id)
    
    rag_chain = (
        {"context": lambda x: context, "question": RunnablePassthrough(), "company_name": lambda x: company_name}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    answer = rag_chain.invoke(user_input)
    
    # Save the interaction to the database
    save_conversation(db_session, session_id, user_input, answer, company_name, job_id)
    
    return answer