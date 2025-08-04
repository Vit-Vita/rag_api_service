import requests
import os
import streamlit as st
import bs4
import datetime
import re
import requests
from urllib.parse import urljoin

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


"""url = "https://www.ricola.com/de-ch/ueber/karriere/offene-stellen/"
output_filename = "ricola_output.html"

print(f"Fetching HTML from {url}...")
try:
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Success! Raw HTML saved to '{output_filename}'")
    print(f"Now, open this file in a text editor or browser and search for 'HSTableLinkSubTitle' or any job title.")

except requests.RequestException as e:
    print(f"Error fetching page: {e}")"""

# Define general company info URLs and combine with job URLs
general_info_urls = [
        "https://www.ricola.com/de-ch/uber/karriere",
        "https://www.ricola.com/de-ch/uber/unternehmen/unsere-geschichte",
        "https://www.ricola.com/de-ch/uber/unternehmen/unsere-werte"
    ]
#all_urls_to_load = general_info_urls + scraped_job_urls

# Load all documents and tag them with metadata
loader = WebBaseLoader(general_info_urls)
all_docs = loader.load()

print(all_docs)