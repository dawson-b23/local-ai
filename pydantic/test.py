from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client
import os

SUPABASE_URL="http://localhost:8000"
SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiIsImlzcyI6InN1cGFiYXNlIiwiaWF0IjoxNzUwMzE2NDAwLCJleHAiOjE5MDgwODI4MDB9.4HKpi7J5zX_eavvlWmvbE0U54v66nldDcS3Kqt87NUI"
OLLAMA_URL="http://localhost:11434"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url=OLLAMA_URL)
vector_store = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name="documents", query_name="match_documents")
results = vector_store.as_retriever().invoke("scope meeting")
print(results)
