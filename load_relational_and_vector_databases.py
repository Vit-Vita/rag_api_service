from dotenv import load_dotenv
load_dotenv()

import os
import sys
import pandas as pd
from tqdm import tqdm

# LangChain components
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader

# import database components
from database import Base, engine, SessionLocal, Company, Job

# --- Credentials and Global Set-ups ---
# OpenAI API key and embedding model
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
PG_CONNECTION_STRING = engine.url.render_as_string(hide_password=False).replace('+psycopg2', '')


# File paths
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/jobs_data/jobs_clean.csv"

# Company Configuration
COMPANIES_CONFIG = {
    "LUKB": {
        "csv_name": "luzerner kantonalbank ag",
        "urls": [
            "https://www.lukb.ch/ueber-uns/jobs/offene-stellen",
            "https://www.lukb.ch/ueber-uns/jobs/trainee",
            "https://www.lukb.ch/ueber-uns/jobs/quereinstieg-kundenberatung",
            "https://www.lukb.ch/ueber-uns/jobs/lukb-als-arbeitgeberin",
            "https://www.lukb.ch/ueber-uns/portraet/nachhaltigkeit"
        ],
        "pdfs": [],
        "general_collection_name": "lukb_general_info",
        "jobs_collection_name": "lukb_job_descriptions"
    },
    "Hager": {
        "csv_name": "hager",
        "urls": ["https://hager.com/de-ch/ueber-uns/arbeiten-bei-hager"],
        "pdfs": [
            "data/hager_pdfs/Firmenpraesentation_Hager_Schweiz_Feb2025_DE.pdf",
            "data/hager_pdfs/Personalreglement_final.pdf",
            "data/hager_pdfs/Übersichtsblatt Sozialversicherungen 2025.pdf"
        ],
        "general_collection_name": "hager_general_info",
        "jobs_collection_name": "hager_job_descriptions"
    }
}

# --- Create Database Tables --
print("Attempting to create tables from the central database models...")
with engine.connect() as connection:
    Base.metadata.create_all(bind=connection)
    connection.commit()
print("Tables created and committed.")

# Use the SessionLocal from database.py for consistency
Session = SessionLocal

# --- Main Ingestion Logic ---
def main():
    # Read CSV
    try:
        df_main = pd.read_csv(CSV_PATH, dtype=str).fillna("")
        print(f"CSV used: {os.path.abspath(CSV_PATH)}")
    except FileNotFoundError:
        sys.exit(f"Error: CSV file not found at {CSV_PATH}")

    # Process each company for relational data
    for company_key, config in COMPANIES_CONFIG.items():
        print(f"\n--- Processing relational data for: {company_key} ---")
        df_company = df_main[df_main["name"].str.lower().str.contains(config["csv_name"])].copy()
        df_company.drop_duplicates(subset=['title'], inplace=True)
        
        if df_company.empty:
            print("Warning: No job data found. Skipping.")
            continue

        session = Session()
        try:
            for _, row in tqdm(df_company.iterrows(), total=len(df_company), desc=f"Populating DB"):
                company_name = row.get("name")
                if not company_name: continue

                company = session.query(Company).filter_by(Name=company_name).first()
                if not company:
                    company = Company(Name=company_name, Addresses=row.get("addresses", ""), Country=row.get("country", ""))
                    session.add(company)
                    session.flush()

                try: job_id = int(row.get("job_id"))
                except (ValueError, TypeError): continue

                job = session.query(Job).filter_by(JobID=job_id).first()
                if not job:
                    job = Job(
                        JobID=job_id, CompanyID=company.CompanyID, Title=row.get("title", ""),
                        Description=row.get("description", ""), Location=row.get("location", ""),
                        Workload=row.get("Workload")
                    )
                    session.add(job)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error populating database: {e}")
        finally:
            session.close()

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    # Process each company for vector data
    for company_key, config in COMPANIES_CONFIG.items():
        print(f"\n--- Ingesting vector data for: {company_key} ---")
        df_company = df_main[df_main["name"].str.lower().str.contains(config["csv_name"])].copy()
        df_company.drop_duplicates(subset=['title'], inplace=True)

        # Ingest general info
        all_general_docs = []
        for url in config["urls"]:
            try:
                loader = WebBaseLoader(url)
                all_general_docs.extend(loader.load())
            except Exception as e: print(f"Warning: Could not load {url} - {e}")
        for pdf_path in config["pdfs"]:
            try:
                if os.path.exists(pdf_path):
                    loader = PyPDFLoader(pdf_path)
                    all_general_docs.extend(loader.load())
                else: print(f"Warning: PDF not found at {pdf_path}")
            except Exception as e: print(f"Warning: Could not load PDF {pdf_path} - {e}")
        
        if all_general_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
            general_splits = text_splitter.split_documents(all_general_docs)
            PGVector.from_documents(
                embedding=embeddings, documents=general_splits,
                collection_name=config["general_collection_name"],
                connection_string=PG_CONNECTION_STRING,
                pre_delete_collection=True
            )
            print(f"PGVector store '{config['general_collection_name']}' created/updated.")

        # Ingest job descriptions
        if not df_company.empty:
            job_documents = [Document(
                page_content=f"Job Title: {row.get('title', '')}\nDescription: {row.get('description', '')}",
                metadata={"job_id": str(row.get("job_id", "")), "title": row.get("title", ""), "company_name": row.get("name", "")}
            ) for _, row in df_company.iterrows()]

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
            job_splits = text_splitter.split_documents(job_documents)
            PGVector.from_documents(
                embedding=embeddings, documents=job_splits,
                collection_name=config["jobs_collection_name"],
                connection_string=PG_CONNECTION_STRING,
                pre_delete_collection=True
            )
            print(f"PGVector store '{config['jobs_collection_name']}' created/updated.")

    print("\n\nData ingestion into PostgreSQL and PGVector completed for all companies.")

if __name__ == "__main__":
    main()