import os
import streamlit as st
import bs4
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
import datetime
import re
from rapidfuzz import process
from dotenv import load_dotenv         
load_dotenv()

# --- Initialize all st.session_state variables FIRST ---
# This block MUST be at the very top of your script, right after imports and loading .env.
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
if "all_jobs" not in st.session_state:
    st.session_state.all_jobs = [] 
if "job_info_shown" not in st.session_state:
    st.session_state.job_info_shown = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "all_jobs_loaded_from_pinecone" not in st.session_state:
    st.session_state.all_jobs_loaded_from_pinecone = False


# --- Function to log unanswered questions (cloud-friendly) ---
def save_unanswered_question(question: str):
    """Log the unanswered question to console."""
    timestamp = datetime.datetime.now().isoformat()
    print(f"UNANSWERED QUESTION: {timestamp} - {question}")

# --- Define and Cache Static RAG Resources ---
@st.cache_resource(show_spinner="Loading RAG models and data...")
def load_static_rag_resources():
    urls = [
        "https://www.lukb.ch/ueber-uns/jobs/offene-stellen",
        "https://www.lukb.ch/ueber-uns/jobs/trainee",
        "https://www.lukb.ch/ueber-uns/jobs/quereinstieg-kundenberatung",
        "https://www.lukb.ch/ueber-uns/jobs/lukb-als-arbeitgeberin",
        "https://www.lukb.ch/ueber-uns/portraet/nachhaltigkeit"
    ]
    
    all_docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        loaded_docs = loader.load()
        all_docs.extend(loaded_docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    splits = text_splitter.split_documents(all_docs)
    
    embeddings = OpenAIEmbeddings()
    
    persist_directory = "chroma_db_lukb"
    vectorstore = Chroma.from_documents(
        splits,
        embeddings,
        persist_directory=persist_directory
    )
    
    pinecone_vectorstore = PineconeStore.from_existing_index(
        index_name=os.getenv("PINECONE_INDEX"),
        embedding=embeddings,
        namespace=os.getenv("PINECONE_NS", "jobs"),
        text_key="description"
    )
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    return vectorstore, pinecone_vectorstore, llm, embeddings

# --- Load Static RAG Resources (executed once due to caching) ---
vectorstore, pinecone_vectorstore, llm, embeddings = load_static_rag_resources()

# Handle cases where resource loading might fail (e.g., missing API keys)
if vectorstore is None or pinecone_vectorstore is None or llm is None or embeddings is None:
    st.error("Failed to load RAG components. Please check your API keys and configuration.")
    st.stop()

# --- Populate st.session_state.all_jobs once per session after resources are loaded ---
if not st.session_state.all_jobs_loaded_from_pinecone:
    try:
        if pinecone_vectorstore:
            st.session_state.all_jobs = pinecone_vectorstore.similarity_search("", k=50)
            st.session_state.all_jobs_loaded_from_pinecone = True
        else:
            st.error("Pinecone vector store not initialized. Cannot load job list.")
            st.session_state.all_jobs_loaded_from_pinecone = True
    except Exception as e:
        st.error(f"Failed to load job list from Pinecone: {e}")
        st.session_state.all_jobs_loaded_from_pinecone = True


# --- Dynamic Context Building Function ---
# This function receives selected_job_id as an explicit argument.
def get_combined_context(question: str,
                         selected_job_id: str, # <-- Explicitly passed now
                         chroma_vectorstore_instance: Chroma,
                         pinecone_vectorstore_instance: PineconeStore) -> str:
    
    chroma_retriever = chroma_vectorstore_instance.as_retriever(search_kwargs={"k": 5})
    general_docs = chroma_retriever.get_relevant_documents(question)
    
    job_docs = []
    
    if selected_job_id:
        job_retriever = pinecone_vectorstore_instance.as_retriever(
            search_kwargs={"k": 1, "filter": {"job_id": selected_job_id}}
        )
        job_docs.extend(job_retriever.get_relevant_documents(question))
    else:
        job_retriever = pinecone_vectorstore_instance.as_retriever(search_kwargs={"k": 10})
        job_docs.extend(job_retriever.get_relevant_documents(question))

    if not job_docs and st.session_state.all_jobs:
        job_docs = st.session_state.all_jobs[:5]
    elif not job_docs:
        job_docs = pinecone_vectorstore_instance.similarity_search("", k=5)

    parts = []
    if job_docs:
        parts.append("Offene Stellen (aus Pinecone VectorDB):\n" + "\n\n".join(d.page_content for d in job_docs))
    if general_docs:
        parts.append("\n\n".join(d.page_content for d in general_docs))
    
    return "\n\n".join(parts)


# Define the prompt template (static)
prompt_template = """Beantworte die folgende Frage basierend auf dem bereitgestellten Kontext.

Kontext:
{context}

Frage:
{question}
Antwort auf Deutsch oder Englisch je nach dem, in welcher Sprache die Frage gestellt wurde.
Frage in Englisch = Antwort in Englisch.
Frage in Deutsch = Antwort in Deutsch.
Beziehen Sie sich bei Ihren Antworten immer auf den angegebenen Kontext als Dokumentation"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)


# This function now ACCEPTS the selected_job_id directly
def get_rag_chain_with_dynamic_context(llm_instance, chroma_vectorstore_instance, pinecone_vectorstore_instance, current_selected_job_id):
    
    # This runnable uses the 'current_selected_job_id' passed to the outer function
    def _dynamic_context_runnable(input_dict):
        question = input_dict["question"]
        # Now, selected_job_id is directly from the argument, not st.session_state
        return get_combined_context(
            question,
            current_selected_job_id, # <-- Use the passed argument
            chroma_vectorstore_instance,
            pinecone_vectorstore_instance
        )

    chain = (
        {"context": _dynamic_context_runnable, "question": RunnablePassthrough()}
        | prompt
        | llm_instance
        | StrOutputParser()
    )
    return chain

# IMPORTANT: The global 'rag_chain' is now removed.
# The chain will be created dynamically when needed to ensure 'selected_job_id' is fresh.

# --- Generate Response Function ---
def generate_response(user_input: str, current_selected_job_id: str) -> str: # <-- Add selected_job_id here
    try:
        # Create the chain just before invoking, ensuring it has the current selected_job_id
        # This guarantees the correct context even if run in a different thread/context.
        rag_chain_for_invoke = get_rag_chain_with_dynamic_context(
            llm, vectorstore, pinecone_vectorstore, current_selected_job_id
        )
        return rag_chain_for_invoke.invoke({"question": user_input})
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I encountered an critical error when trying to generate a response."

# --- Main Streamlit App UI ---
st.title("Chatbot Interface (LuKB RAG)")

def show_job_menu():
    st.markdown("### Aktuell offene Stellen")
    if not st.session_state.all_jobs:
        st.info("No jobs available to display.")
        return

    for i, doc in enumerate(st.session_state.all_jobs, 1):
        meta = doc.metadata
        job_title = meta.get("title", "")
        job_id = meta.get("job_id", f"no_id_{i}")
        
        label = f"**{job_title}** \nOrt: {meta.get('location', '?')} – Pensum: {meta.get('workload', '?')}"
        if st.button(label, key=f"job_btn_{job_id}"):
            st.session_state.selected_job = job_id
            st.session_state.job_info_shown = False
            st.session_state.messages.append({"role": "assistant", "content": f"✅ **{job_title}** ausgewählt. Stellen Sie Ihre Frage dazu.\n_Wenn Sie Fragen zu anderen Stellen haben, schreiben Sie **exit**._"})
            st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Nachricht eingeben …")

# --- Handle user input ─────────────────────────────────────────────────
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if user_input.strip().lower() == "exit":
        st.session_state.selected_job = None
        st.session_state.job_info_shown = False
        response_msg = "🔄 Job-Auswahl zurückgesetzt. Bitte wählen Sie eine andere Stelle:"
        st.chat_message("assistant").markdown(response_msg)
        st.session_state.messages.append({"role": "assistant", "content": response_msg})
        st.rerun()
    elif st.session_state.selected_job is None:
        titles = [d.metadata.get("title", "") for d in st.session_state.all_jobs]
        ids = [d.metadata.get("job_id", "") for d in st.session_state.all_jobs]

        picked_idx = None
        if user_input.strip() in ids:
            picked_idx = ids.index(user_input.strip())
        else:
            match, score, fuzzy_idx = process.extractOne(
                user_input, titles, score_cutoff=70
            ) or (None, None, None)
            if fuzzy_idx is not None:
                picked_idx = fuzzy_idx

        if picked_idx is None:
            response_msg = "❌ Ich konnte diese Stelle nicht finden. Bitte geben Sie den exakten Titel oder die Job-ID an."
            st.chat_message("assistant").markdown(response_msg)
            st.session_state.messages.append({"role": "assistant", "content": response_msg})
        else:
            picked_doc = st.session_state.all_jobs[picked_idx]
            st.session_state.selected_job = picked_doc.metadata["job_id"]
            st.session_state.job_info_shown = False
            response_msg = (
                f"✅ **{picked_doc.metadata['title']}** ausgewählt. "
                "Stellen Sie Ihre Frage dazu.\n"
                "_Wenn Sie Fragen zu anderen Stellen haben, schreiben Sie **exit**._"
            )
            st.chat_message("assistant").markdown(response_msg)
            st.session_state.messages.append({"role": "assistant", "content": response_msg})
            st.rerun()
    else:
        job_id = st.session_state.selected_job
        picked_doc = next((doc for doc in st.session_state.all_jobs if doc.metadata.get("job_id") == job_id), None)

        if picked_doc and not st.session_state.get('job_info_shown', False):
            job_info_message = (
                f"📌 **Ausgewählte Stelle:**\n"
                f"- **Titel:** {picked_doc.metadata.get('title', '')}\n"
                f"- **Ort:** {picked_doc.metadata.get('location', '')}\n"
                f"- **Pensum:** {picked_doc.metadata.get('workload', '?')}\n"
            )
            st.chat_message("assistant").markdown(job_info_message)
            st.session_state.messages.append({"role": "assistant", "content": job_info_message})
            st.session_state.job_info_shown = True

        with st.spinner("Thinking..."):
            # Pass the current selected_job_id from session state to generate_response
            response = generate_response(user_input.replace("\n", " "), st.session_state.selected_job)
        
        if "ich weiß nicht." in response.lower():
            save_unanswered_question(user_input)
            
        final_response_display = response + "\n\n*Wenn Sie Fragen zu anderen Stellen haben, schreiben Sie **exit***."
        st.chat_message("assistant").markdown(final_response_display)
        st.session_state.messages.append({"role": "assistant", "content": response})

if st.session_state.selected_job is None:
    show_job_menu()