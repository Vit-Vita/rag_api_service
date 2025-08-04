#!/usr/bin/env python
"""
load_vectors.py  path/to/jobs_clean.csv

• Deletes all vectors in the configured Pinecone namespace
• Reads the CSV produced by Apache Hop
• Embeds each row with OpenAI embeddings
• Upserts in batches of 100
"""

# ───── 0. Imports & helpers ──────────────────────────────────────────
from dotenv import load_dotenv        
load_dotenv()

import os, sys, time, pandas as pd
from tqdm import tqdm
import openai
from pinecone import Pinecone, ServerlessSpec  
from pinecone.core.client.exceptions import NotFoundException

# ───── 1. Environment config ────────────────────────────────────────
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME       = os.getenv("PINECONE_INDEX", "jobs")
NAMESPACE        = os.getenv("PINECONE_NS", "jobs")
REGION           = os.getenv("PINECONE_REGION", "us-east1-gcp")   # adjust if needed

openai.api_key   = os.environ["OPENAI_API_KEY"]
EMBED_MODEL      = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

CSV_PATH   = sys.argv[1] if len(sys.argv) > 1 else "jobs_data/jobs_clean.csv"
BATCH_SIZE = 100
EMBED_DIM  = 1536                      # dim for text-embedding-3-small

# ───── 2. Read the CSV ──────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
if "job_id" not in df.columns:
    sys.exit("  Column 'job_id' is required. Check your CSV.")

print("CSV used        :", os.path.abspath(CSV_PATH))
print("Rows read       :", len(df))
print("First 5 names   :", df["name"].head().tolist())


df = df[df["name"].str.lower() == "luzerner kantonalbank ag"]
df = df.drop_duplicates(subset=['title'])  #####

#print("Rows after filter:", len(df))
#print("Unique names kept:", df["name"].unique()[:5])


# ───── 3. Connect to (or create) the Pinecone index ────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)


index = pc.Index(INDEX_NAME)

# ───── 4. Wipe the old namespace ────────────────────────────────────
print(f"Deleting all vectors in namespace '{NAMESPACE}' …", flush=True)
try:
    index.delete(delete_all=True, namespace=NAMESPACE)
except NotFoundException:
    print(f"Namespace '{NAMESPACE}' didn’t exist yet; skip delete.")


# Optional: wait until the namespace is empty (large indexes only)
while index.describe_index_stats()\
          .namespaces.get(NAMESPACE, {})\
          .get("vectorCount", 0):
    time.sleep(1)

# ───── 5. Embedding helper ──────────────────────────────────────────
def embed(text: str) -> list[float]:
    # v1.0+ endpoint
    resp = openai.embeddings.create(
        model=EMBED_MODEL,
        input=text            # str or list[str] both work
    )
    return resp.data[0].embedding

# ───── 6. Batch-upsert new vectors ─────────────────────────────────
batch = []
for _, row in tqdm(df.iterrows(), total=len(df), desc=" Upserting"):
    batch.append((
        str(row["job_id"]),                          # vector ID
        embed(f"{row.get('title', '')} {row.get('description', '')}"),
        row.to_dict()                                # metadata
    ))

    if len(batch) == BATCH_SIZE:
        index.upsert(vectors=batch, namespace=NAMESPACE)
        batch.clear()

if batch:                                           # final remainder
    index.upsert(vectors=batch, namespace=NAMESPACE)

print("  Vector store refreshed.")
