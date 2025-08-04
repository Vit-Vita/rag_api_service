import os
import streamlit as st
import bs4
import datetime
import re
import requests
from urllib.parse import urljoin
import shutil
from langchain.output_parsers import PydanticOutputParser

# LangChain and vector store imports
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from rapidfuzz import process
from dotenv import load_dotenv
from langchain_community.document_loaders import RecursiveUrlLoader
load_dotenv()
from langchain.docstore.document import Document
from pydantic import BaseModel, Field
from typing import Optional


# --- Function to save unanswered questions to a text file
def save_unanswered_question(question: str):
    """Append the unanswered question along with a timestamp to a text file."""
    timestamp = datetime.datetime.now().isoformat()
    entry = f"{timestamp} - {question}\n"
    with open("Unanswered_questions_Ricola.txt", "a", encoding="utf-8") as f:
        f.write(entry)

# --- Initialize st.session_state variables
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
if "all_jobs" not in st.session_state:
    st.session_state.all_jobs = []
if "job_info_shown" not in st.session_state:
    st.session_state.job_info_shown = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "jobs_loaded" not in st.session_state:
    st.session_state.jobs_loaded = False


# Pydantic class
class JobDetails(BaseModel):
    """Structured data about a job posting."""
    title: str = Field(description="The exact and complete title of the job position.")
    workload: Optional[str] = Field(description="The workload or 'Pensum', e.g., '80-100%'. If not found, this should be null.")


# RAG Chain 
@st.cache_resource(show_spinner="Loading company info and job posts...")

def load_rag_components():
    persist_directory = "chroma_db_ricola"
    if os.path.exists(persist_directory):
        print(f"Deleting old database directory: {persist_directory}")
        shutil.rmtree(persist_directory)
        print("Old database deleted successfully.")
    # --- Part 1: Load specific job posts with WebBaseLoader ---
    job_post_urls = [
        "https://career.ricola.com/Vacancies/1014/Description/1",
        "https://career.ricola.com/Vacancies/1016/Description/1",
        "https://career.ricola.com/Vacancies/1015/Description/1",
        "https://career.ricola.com/Vacancies/1013/Description/1",
        "https://career.ricola.com/Vacancies/1012/Description/1",
        "https://career.ricola.com/Vacancies/1010/Description/1",
        "https://career.ricola.com/Vacancies/1009/Description/1",
        "https://career.ricola.com/Vacancies/960/Description/1",
        "https://career.ricola.com/Vacancies/997/Description/1"
        ]
    job_docs = []
    print(f"Loading {len(job_post_urls)} specific job posts with targeted extraction...")
    
    for url in job_post_urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = bs4.BeautifulSoup(response.content, 'lxml')
            
            content_text = soup.get_text(strip=True, separator=" ")

            # Create a LangChain Document manually with the full page text
            doc = Document(
                page_content=content_text,
                metadata={"source": url}
            )
            job_docs.append(doc)

        except Exception as e:
            print(f"  - Failed to load {url}. Error: {e}")
    
    #job_docs = WebBaseLoader(job_post_urls).load()
    general_info_start_url = "https://www.ricola.com/de-ch/uber/"
    recursive_loader = RecursiveUrlLoader(url=general_info_start_url, max_depth=2, prevent_outside=True, extractor=lambda html: bs4.BeautifulSoup(html, "lxml").get_text())
    general_docs = recursive_loader.load()
    all_docs = general_docs + job_docs
    unique_docs_map = {doc.page_content: doc for doc in all_docs}
    all_docs = list(unique_docs_map.values())
    print(f"Total unique documents to process: {len(all_docs)}")

    # --- Part 4: Tag documents and use LLM for job details ---
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # +++ THIS IS THE ALTERNATIVE METHOD +++
    # 1. Create a parser instance from your Pydantic class
    parser = PydanticOutputParser(pydantic_object=JobDetails)

    # 2. Create a prompt template that includes the formatting instructions from the parser
    prompt_for_extraction = PromptTemplate(
        template="From the following text from a job posting, extract the required information.\n{format_instructions}\nTEXT: ```{text}```",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 3. Create the extraction chain
    extraction_chain = prompt_for_extraction | llm | parser
    
    for doc in all_docs:
        source_url = doc.metadata.get('source', '')
        if source_url in job_post_urls:
            doc.metadata['type'] = 'job'
            print(f"  > Using LLM to parse job details from: {source_url}")
            try:
                page_text = bs4.BeautifulSoup(doc.page_content, 'lxml').get_text()
                
                # Invoke the chain with the text
                extracted_data = extraction_chain.invoke({"text": page_text[:4000]})
                
                title = extracted_data.title if extracted_data.title else "Unknown Job Title"
                workload = extracted_data.workload if extracted_data.workload else "N/A"
                print(f"    - LLM extracted Title: {title}")

                doc.metadata['title'] = title
                doc.metadata['job_id'] = title
                doc.metadata['workload'] = workload

            except Exception as e:
                print(f"    - LLM extraction failed for {source_url}. Error: {e}")
                doc.metadata['title'] = "Unknown Job Title (Parsing Failed)"
                doc.metadata['job_id'] = source_url
                doc.metadata['workload'] = "N/A"
        else:
            doc.metadata['type'] = 'info'

    # --- The rest of the function remains the same ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    splits = text_splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="chroma_db_ricola"
    )

    return vectorstore, llm, embeddings


# Load Static RAG Resources (executed once due to caching)
vectorstore, llm, embeddings = load_rag_components()

if vectorstore and not st.session_state.jobs_loaded:
    # 1. Get the potentially duplicated list from the database
    # We set k to a high number to make sure we retrieve everything.
    all_retrieved_jobs = vectorstore.similarity_search("", k=100, filter={"type": "job"})
    print(f"Initially retrieved {len(all_retrieved_jobs)} job documents from ChromaDB.")

    # 2. Force the list to be unique based on the job title (which is the job_id)
    unique_jobs = {}
    for doc in all_retrieved_jobs:
        # The job_id is the title extracted by the LLM
        job_id = doc.metadata.get("job_id")
        if job_id and job_id not in unique_jobs:
            unique_jobs[job_id] = doc
    
    # 3. Assign the clean, unique list of documents to the session state
    st.session_state.all_jobs = list(unique_jobs.values())
    st.session_state.jobs_loaded = True
    # +++ DEBUGGING: Print the content of each final job document +++
    print("\n--- Content of Final Unique Job Documents ---")
    for i, doc in enumerate(st.session_state.all_jobs):
        title = doc.metadata.get('title', 'NO TITLE')
        print(f"\n----- JOB {i+1}: {title} -----")
        # Print the first 500 characters of the content to keep the log clean
        print(doc.page_content[:1500] + "...")
        print("-------------------------------------------------")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    # This print statement should now show the correct number (e.g., 9)
    print(f"Loaded {len(st.session_state.all_jobs)} unique jobs into the UI menu.")


# Dynamic Context Building Function
def get_combined_context(question: str, selected_job_id: str, chroma_vectorstore_instance: Chroma) -> str:
    parts = []
    
    # 1. Always retrieve general company information relevant to the user's question
    general_retriever = chroma_vectorstore_instance.as_retriever(
        search_kwargs={"k": 3, "filter": {"type": "info"}}
    )
    general_docs = general_retriever.get_relevant_documents(question)
    if general_docs:
        parts.append("Allgemeine Unternehmensinformationen:\n" + "\n\n".join(d.page_content for d in general_docs))

    # 2. If a specific job is selected, add its full text to the context
    if selected_job_id:
        # The job_id is the title, so we filter by that.
        job_retriever = chroma_vectorstore_instance.as_retriever(
            search_kwargs={"k": 1, "filter": {"job_id": selected_job_id}}
        )
        # We search with a generic query because the filter is what matters
        job_docs = job_retriever.get_relevant_documents("job description")
        if job_docs:
            parts.append("Informationen zur ausgewählten Stelle:\n" + "\n\n".join(d.page_content for d in job_docs))

    final_context = "\n\n---\n\n".join(parts)
    # If no context was found at all, provide a fallback message
    return final_context if final_context else "Kein spezifischer Kontext gefunden."

# Define the prompt template (static)
prompt_template = """
Sie sind ein hilfreicher Assistent, der Fragen von potenziellen Bewerbern für eine Stelle bei Ricola beantwortet.
Ihr Ziel ist es nicht nur, die Fragen zu beantworten, sondern auch die Soft Skills, persönlichen Werte und Arbeitswerte des Bewerbers durch gezielte Follow-up-Fragen zu bewerten.

Kontext: {context}
Frage: {question}

Anweisungen:
1. Beantworten Sie die Frage basierend auf dem bereitgestellten Kontext.
2. Stellen Sie sicher, dass Ihre Antworten klar und präzise sind.
3. Antworten Sie auf Deutsch.
4. Fügen Sie am Ende Ihrer Antwort eine relevante Follow-up-Frage ein, die auf die Antwort des Bewerbers Bezug nimmt und darauf abzielt, dessen Soft Skills oder Motivation zu bewerten.

Ihre Antwort:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)


# This function defines the full RAG chain, including the dynamic context
def get_rag_chain_with_dynamic_context(llm_instance, chroma_vectorstore_instance, current_selected_job_id):
    def _dynamic_context_runnable(input_dict):
        question = input_dict["question"]
        return get_combined_context(
            question,
            current_selected_job_id,
            chroma_vectorstore_instance
        )

    chain = (
        {"context": _dynamic_context_runnable, "question": RunnablePassthrough()}
        | prompt
        | llm_instance
        | StrOutputParser()
    )
    return chain

# Generate Response
def generate_response(user_input: str, current_selected_job_id: str) -> str:
    try:
        rag_chain_for_invoke = get_rag_chain_with_dynamic_context(
            llm, vectorstore, current_selected_job_id
        )
        return rag_chain_for_invoke.invoke({"question": user_input})
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main Streamlit App UI
st.title("Ricola Karriere-Assistent")
def show_job_menu():
    st.markdown("### Aktuell offene Stellen")
    if not st.session_state.all_jobs:
        st.info("Zurzeit sind keine offenen Stellen verfügbar oder sie konnten nicht geladen werden.")
        return

    # We use enumerate to get a safe, unique index 'i' for the button key
    for i, doc in enumerate(st.session_state.all_jobs):
        meta = doc.metadata
        job_title = meta.get("title", "Unbekannte Stelle")
        job_id = meta.get("job_id", job_title) # The ID is now the title
        workload = meta.get("workload", "?") # Get the new workload metadata

        # NEW: Updated label to include the workload
        label = f"**{job_title}** – Pensum: {workload}"
        
        # We still use the index 'i' for the key to prevent errors
        if st.button(label, key=f"job_btn_{i}"):
            st.session_state.selected_job = job_id # Select the job by its title
            st.session_state.job_info_shown = False
            st.session_state.messages.append({"role": "assistant", "content": f"✅ **{job_title}** ausgewählt. Wozu möchten Sie mehr erfahren?\n_Um die Auswahl zurückzusetzen, schreiben Sie **exit**._"})
            st.rerun()

# Chat UI Logic
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Fragen Sie etwas über Ricola oder eine Stelle..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if user_input.strip().lower() == "exit":
        st.session_state.selected_job = None
        st.session_state.job_info_shown = False
        response_msg = "🔄 Job-Auswahl zurückgesetzt. Bitte wählen Sie eine andere Stelle oder stellen Sie eine allgemeine Frage."
        st.chat_message("assistant").markdown(response_msg)
        st.session_state.messages.append({"role": "assistant", "content": response_msg})
        st.rerun()

    else:
        # When a job is selected, show its title as confirmation
        if st.session_state.selected_job and not st.session_state.job_info_shown:
            job_id = st.session_state.selected_job
            picked_doc = next((doc for doc in st.session_state.all_jobs if doc.metadata.get("job_id") == job_id), None)
            if picked_doc:
                job_info_message = (f"📌 **Informationen zu:** {picked_doc.metadata.get('title', '')}")
                st.chat_message("assistant").markdown(job_info_message)
                st.session_state.messages.append({"role": "assistant", "content": job_info_message})
                st.session_state.job_info_shown = True

        with st.spinner("Thinking..."):
            response = generate_response(user_input, st.session_state.selected_job)

        if "ich weiß nicht" in response.lower() or "kann ich ihnen nicht beantworten" in response.lower():
             save_unanswered_question(user_input)

        final_response_display = response
        if st.session_state.selected_job:
            final_response_display += "\n\n*Um die Auswahl zurückzusetzen, schreiben Sie **exit**.*"

        st.chat_message("assistant").markdown(final_response_display)
        st.session_state.messages.append({"role": "assistant", "content": response})

if st.session_state.selected_job is None:
    show_job_menu()