import os
import streamlit as st
import bs4  # For completeness if needed by loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
# --- NEW: Import the PDF loader ---
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore
import datetime
import re
from rapidfuzz import process
from dotenv import load_dotenv        
load_dotenv()

# --- Function to save unanswered questions to a text file ---
def save_unanswered_question(question: str):
    """Append the unanswered question along with a timestamp to Unanswered_questions_HAGER.txt."""
    timestamp = datetime.datetime.now().isoformat()
    entry = f"{timestamp} - {question}\n"
    with open("Unanswered_questions_HAGER.txt", "a", encoding="utf-8") as f:
        f.write(entry)

# --- Define and Cache the RAG Chain ---
@st.cache_resource(show_spinner=True)
def load_rag_chain():
    # 1. Load documents from the HAGER website
    loader_web = WebBaseLoader("https://hager.com/de-ch/ueber-uns/arbeiten-bei-hager")
    web_docs = loader_web.load()
    
    # 2. Load documents from the three PDFs
    # Ensure the file paths below are correct relative to your working directory.
    pdf_paths = [
        "pdfs/Firmenpraesentation_Hager_Schweiz_Feb2025_DE.pdf",
        "pdfs/Personalreglement_final.pdf",
        "pdfs/Übersichtsblatt Sozialversicherungen 2025.pdf"
    ]
    pdf_docs = []
    for path in pdf_paths:
        loader_pdf = PyPDFLoader(path)
        pdf_docs.extend(loader_pdf.load())
    
    # 3. Combine web and PDF documents
    docs = web_docs + pdf_docs
    
    # 4. Split the combined documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 5. Create the Chroma vector store from the document chunks
    persist_directory = "chroma_db_HAGER"
    vectorstore = Chroma.from_documents(
        splits,
        OpenAIEmbeddings(),
        persist_directory=persist_directory
    )
    st.session_state["vectorstore"] = vectorstore
    
    pinecone_vectorstore = PineconeStore.from_existing_index(
        index_name=os.getenv("PINECONE_INDEX"),          # comes from .env
        embedding=OpenAIEmbeddings(),
        namespace=os.getenv("PINECONE_NS", "jobs"),
        text_key="description"       # same you used in loader
    )
    jobs_retriever = pinecone_vectorstore.as_retriever(search_kwargs={"k": 10})

    st.session_state["pinecone_vectorstore"] = pinecone_vectorstore

    # --- Keep the existing Chroma retriever for the PDFs & website ---
    retriever = vectorstore.as_retriever()

    # --- Helper that merges both retrievers and formats them ---
    def build_context(question: str) -> str:
        """Merge general knowledge with job-postings context.

        • The chunks coming from Pinecone represent the **open jobs**.
        • If the similarity search returns no job docs, fall back to the
          first 20 vectors so the user always sees the openings list.
        """
        general_docs = retriever.get_relevant_documents(question)
        job_docs     = jobs_retriever.get_relevant_documents(question)

        # Fallback: bring in a snapshot of all open jobs (k=20)  
        if not job_docs:
            job_docs = pinecone_vectorstore.similarity_search("", k=20)

        parts = []
        if job_docs:
            parts.append(
                "Offene Stellen (aus Pinecone VectorDB):\n"
                + "\n\n".join(d.page_content for d in job_docs)
            )
        if general_docs:
            parts.append("\n\n".join(d.page_content for d in general_docs))

        return "\n\n".join(parts)
    
    # 6. Define the prompt template (unchanged from your original version)
    prompt_template = """Beantworte die folgende Frage basierend auf dem bereitgestellten Kontext.

Kontext:
{context}

Frage:
{question}

Antwort auf Deutsch oder Englisch je nach Sprache der Frage."""
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    
    # 7. Create the LLM and build the chain
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": build_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

rag_chain = load_rag_chain()

pinecone_vectorstore = st.session_state["pinecone_vectorstore"]
vectorstore          = st.session_state["vectorstore"]            

def generate_response(user_input: str) -> str:
    try:
        return rag_chain.invoke(user_input)
    except Exception as e:
        return f"Error: {str(e)}"

# --- Chat UI & state -----------------------------------------------------------------


if "selected_job" not in st.session_state:          # None means menu phase
    st.session_state.selected_job = None
if "all_jobs" not in st.session_state:              # cache the menu once
    # pull the first 50 postings (empty query = “give me everything”)
    st.session_state.all_jobs = pinecone_vectorstore.similarity_search("", k=50)

def show_job_menu():
    st.markdown("### Aktuell offene Stellen")
    for i, doc in enumerate(st.session_state.all_jobs, 1):
        meta = doc.metadata
        st.markdown(
            f"**{i}. {meta.get('title','')}**  \n"
            f"Ort: {meta.get('location','?')} – Pensum: {meta.get('workload','?')}"
        )
    st.markdown(
        "\n**Zu welcher Stelle haben Sie Fragen?** "
        "_Tippen Sie den **exakten Titel** oder die **Job-ID**._"
    )

show_job_menu_flag = st.session_state.selected_job is None

st.title("Chatbot Interface (HAGER RAG)")

# Initialize chat_history if not already present   ##### ADDED THIS
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history
for chat in st.session_state.get("chat_history", []):
    st.chat_message(chat["role"]).markdown(chat["message"])

user_input = st.chat_input("Nachricht eingeben …")

# ── Handle user input ─────────────────────────────────────────────────
if user_input:
    if user_input.strip().lower() == "exit":
        # reset and show the menu again
        st.session_state.selected_job = None
        st.chat_message("assistant").markdown(
            "🔄 Job-Auswahl zurückgesetzt.\n"
            "Bitte wählen Sie eine andere Stelle:"
        )
        show_job_menu()
        st.session_state.chat_history.append({"role": "assistant",
                                              "message": "🔄 Job-Auswahl zurückgesetzt."})
    elif st.session_state.selected_job is None:
        # menu phase → resolve the user’s pick
        titles = [d.metadata.get("title", "") for d in st.session_state.all_jobs]
        ids    = [d.metadata.get("job_id", "") for d in st.session_state.all_jobs]

        # try exact ID first
        if user_input.strip() in ids:
            picked_idx = ids.index(user_input.strip())
        else:
            # fuzzy-match against titles
            match, score, picked_idx = process.extractOne(
                user_input, titles, score_cutoff=70
            ) or (None, None, None)

        if picked_idx is None:
            st.chat_message("assistant").markdown(
                "❌ Ich konnte diese Stelle nicht finden. "
                "Bitte geben Sie den exakten Titel oder die Job-ID an."
            )
        else:
            picked_doc = st.session_state.all_jobs[picked_idx]
            st.session_state.selected_job = picked_doc.metadata["job_id"]
            st.chat_message("assistant").markdown(
                f"✅ **{picked_doc.metadata['title']}** ausgewählt. "
                "Stellen Sie Ihre Frage dazu.\n"
                "_Wenn Sie Fragen zu anderen Stellen haben, schreiben Sie **exit**._"
            )
    else:
        # deep-dive phase → build filtered retriever for this job
        job_id = st.session_state.selected_job
        jobs_retriever = pinecone_vectorstore.as_retriever(
            search_kwargs={"k": 10, "filter": {"job_id": job_id}}
        )

        def build_context(question: str) -> str:
            general_docs = vectorstore.as_retriever().get_relevant_documents(question)
            job_docs     = jobs_retriever.get_relevant_documents(question)
            if not job_docs:  # safeguard
                job_docs = pinecone_vectorstore.similarity_search(
                    "", k=1, filter={"job_id": job_id}
                )
            parts = [
                "Offene Stelle (Details):\n" +
                "\n\n".join(d.page_content for d in job_docs)
            ]
            if general_docs:
                parts.append("\n\n".join(d.page_content for d in general_docs))
            return "\n\n".join(parts)

        # invoke the chain with per-job context
        response = generate_response(user_input.replace("\n", " "))
        response += "\n\n*Wenn Sie Fragen zu anderen Stellen haben, schreiben Sie **exit***."

        # show & log
        st.chat_message("user").markdown(user_input)
        st.chat_message("assistant").markdown(response)
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        st.session_state.chat_history.append({"role": "assistant", "message": response})

# show menu at top if no job selected yet
if show_job_menu_flag:
    show_job_menu()
