# database.py
from sqlalchemy import create_engine, Column, Integer, Text, Date, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from config import settings

# --- SQLAlchemy Setup ---
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Models ---

class Candidate(Base):
    __tablename__ = 'candidate'
    CandidateID = Column(Integer, primary_key=True, autoincrement=True)
    FirstName = Column(Text)
    LastName = Column(Text)
    DOB = Column(Date)
    Address = Column(Text)
    City = Column(Text)
    Canton = Column(Text)
    Country = Column(Text)
    PostalCode = Column(Text)
    SessionUUID = Column(Text, unique=True, nullable=True)
    Phone = Column(Text)
    Email = Column(Text)
    applications = relationship("Application", back_populates="candidate")
    chats = relationship("Chat", back_populates="candidate")

class Company(Base):
    __tablename__ = 'company'
    CompanyID = Column(Integer, primary_key=True, autoincrement=True)
    Name = Column(Text, unique=True)
    Addresses = Column(Text)
    City = Column(Text)
    Canton = Column(Text)
    Country = Column(Text)
    PostalCode = Column(Text)
    jobs = relationship("Job", back_populates="company")

class Job(Base):
    __tablename__ = 'job'
    JobID = Column(Integer, primary_key=True)
    CompanyID = Column(Integer, ForeignKey('company.CompanyID'))
    Title = Column(Text)
    Description = Column(Text)
    Addresses = Column(Text)
    Location = Column(Text)
    Canton = Column(Text)
    PostalCode = Column(Text)
    Country = Column(Text)
    Created = Column(DateTime)
    Updated = Column(DateTime)
    Homeoffice = Column(Boolean, default=False)
    Temporary = Column(Boolean, default=False)
    Language = Column(Text)
    Education = Column(Text)
    Workload = Column(Text)
    company = relationship("Company", back_populates="jobs")
    ideal_profile = relationship("IdealCandidateProfile", uselist=False, back_populates="job")
    applications = relationship("Application", back_populates="job")
    chats = relationship("Chat", back_populates="job")

class IdealCandidateProfile(Base):
    __tablename__ = 'idealcandidateprofile'
    IdealCandidateID = Column(Integer, primary_key=True, autoincrement=True)
    JobID = Column(Integer, ForeignKey('job.JobID'))
    Experience = Column(Integer)
    Skills = Column(Text)
    Education = Column(Text)
    Languages = Column(Text)
    job = relationship("Job", back_populates="ideal_profile")

class Application(Base):
    __tablename__ = 'applies'
    ApplicationID = Column(Integer, primary_key=True, autoincrement=True)
    CandidateID = Column(Integer, ForeignKey('candidate.CandidateID'))
    JobID = Column(Integer, ForeignKey('job.JobID'))
    ApplicationDate = Column(DateTime)
    Status = Column(Text)
    candidate = relationship("Candidate", back_populates="applications")
    job = relationship("Job", back_populates="applications")


class Chat(Base):
    __tablename__ = 'chat'
    ChatID = Column(Integer, primary_key=True, autoincrement=True)
    JobID = Column(Integer, ForeignKey('job.JobID'), nullable=True)
    CandidateID = Column(Integer, ForeignKey('candidate.CandidateID'), nullable=True) # Kept this for future use
    # Added SessionID required by the API
    SessionID = Column(Text, index=True, nullable=False)
    CreatedAt = Column(DateTime)
    candidate = relationship("Candidate", back_populates="chats")
    job = relationship("Job", back_populates="chats")
    messages = relationship("Message", back_populates="chat")


class Message(Base):
    __tablename__ = 'messages'
    MessageID = Column(Integer, primary_key=True, autoincrement=True)
    ChatID = Column(Integer, ForeignKey('chat.ChatID'))
    # Changed SenderType to Text to store "user" or "ai"
    SenderType = Column(Text)
    Content = Column(Text)
    Time = Column(DateTime)
    chat = relationship("Chat", back_populates="messages")

# --- Dependency for FastAPI ---
def get_db():
    """FastAPI dependency to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()