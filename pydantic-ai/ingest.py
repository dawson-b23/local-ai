import os
import time
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter
from supabase import create_client, Client
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd
import ollama
from datetime import datetime
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
WATCH_DIRECTORY = "/tank/local-ai/data"
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.xlsx', '.csv', '.md'}
IGNORED_EXTENSIONS = {'.STEP', '.iwp', '.jfif'}
EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_HOST = "http://localhost:11434"

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Docling
converter = DocumentConverter()

def generate_embedding(text: str) -> List[float]:
    """Generate embeddings using Ollama."""
    logger.info("Calling generate_embedding()\n")

    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text, options={"host": OLLAMA_HOST})
    return response['embedding']

def split_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks for vector storage."""
    logger.info("Calling split_text()\n")

    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1
        if current_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        logger.info("Calling on_created()\n")
        if event.is_directory:
            return
        self.process_file(event.src_path, "created")

    def on_modified(self, event):
        logger.info("Calling on_modified()\n")
        if event.is_directory:
            return
        self.process_file(event.src_path, "modified")

    def on_deleted(self, event):
        logger.info("Calling on_deleted()\n")
        if event.is_directory:
            return
        self.delete_file(event.src_path)

    def process_file(self, file_path: str, event_type: str):
        logger.info(f"Processing file: {file_path} ({event_type})")
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS or file_ext in IGNORED_EXTENSIONS:
            logger.info(f"Skipping file {file_path}: Invalid extension")
            return

        file_id = file_path
        file_title = os.path.splitext(os.path.basename(file_path))[0]
        is_press20 = 'press' in file_title.lower()

        try:
            self.delete_old_data(file_id, is_press20)
            if is_press20 and file_ext == '.csv':
                logger.info(f"Processing Press20 CSV: {file_path}")
                self.process_press20_csv(file_path, file_id, file_title)
            else:
                logger.info(f"Processing document: {file_path}")
                self.process_document(file_path, file_id, file_title)
        except Exception as e:
            logger.error(f"error processing {file_path}: {str(e)}", exc_info=True)


    def delete_old_data(self, file_id: str, is_press20: bool):
        """Delete existing records for the file."""
        logger.info("Calling delete_old_data()\n")

        supabase.table("documents").delete().eq("metadata->>file_id", file_id).execute()
        supabase.table("document_rows").delete().eq("dataset_id", file_id).execute()
        supabase.table("document_metadata").delete().eq("id", file_id).execute()
        if is_press20:
            supabase.table("press20_data").delete().eq("dataset_id", file_id).execute()

    def process_press20_csv(self, file_path: str, file_id: str, file_title: str):
        """Process press20 CSV files and insert directly into Supabase."""
        logger.info("Calling process_press20_csv()\n")
        # Insert metadata
        supabase.table("document_metadata").upsert({
            "id": file_id,
            "title": file_title,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        # Read CSV
        df = pd.read_csv(file_path)
        schema = json.dumps(list(df.columns))

        # Update schema in metadata
        supabase.table("document_metadata").update({
            "schema": schema
        }).eq("id", file_id).execute()

        # Insert rows into press20_data
        rows = df.to_dict('records')
        for row in rows:
            row_data = {
                "dataset_id": file_id,
                **{k: v for k, v in row.items() if pd.notna(v)}
            }
            supabase.table("press20_data").insert(row_data).execute()

    def process_document(self, file_path: str, file_id: str, file_title: str):
        """Process non-press20 documents using Docling."""
        logger.info("Calling process_document()\n")
        # Insert metadata
        supabase.table("document_metadata").upsert({
            "id": file_id,
            "title": file_title,
            "created_at": datetime.utcnow().isoformat()
        }).execute()


        # TODO: add check to make sure not to parse press20 data 
        
        # Convert document using Docling
        doc = converter.convert(file_path).document
        text = doc.export_to_markdown()

        # Handle tabular data if present (e.g., from Excel/CSV)
        schema = None
        if file_path.endswith(('.xlsx', '.csv')):
            df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
            schema = json.dumps(list(df.columns))
            rows = df.to_dict('records')
            for row in rows:
                row_data = {
                    "dataset_id": file_id,
                    "row_data": json.dumps({k: v for k, v in row.items() if pd.notna(v)})
                }
                supabase.table("document_rows").insert(row_data).execute()

        # Update schema if applicable
        if schema:
            supabase.table("document_metadata").update({
                "schema": schema
            }).eq("id", file_id).execute()

        # Split text and generate embeddings
        chunks = split_text(text)
        for chunk in chunks:
            embedding = generate_embedding(chunk)
            supabase.table("documents").insert({
                "content": chunk,
                "metadata": {"file_id": file_id, "file_title": file_title},
                "embedding": embedding
            }).execute()

    def delete_file(self, file_path: str):
        """Handle file deletion."""
        logger.info("Calling delete_file()\n")
        file_id = file_path
        try:
            self.delete_old_data(file_id, 'press' in os.path.basename(file_path).lower())
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")

def setup_database():
    """Set up database tables and functions."""
    logger.info("Calling setup_database()\n")
    queries = [
        """
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS document_metadata (
            id TEXT PRIMARY KEY,
            title TEXT,
            url TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            schema TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS document_rows (
            id SERIAL PRIMARY KEY,
            dataset_id TEXT REFERENCES document_metadata(id),
            row_data JSONB
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS documents (
            id BIGSERIAL PRIMARY KEY,
            content TEXT,
            metadata JSONB,
            embedding VECTOR(768)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS press20_data (
            id SERIAL PRIMARY KEY,
            shot_num INTEGER,
            overallPassFail TEXT,
            dataset_id TEXT REFERENCES document_metadata(id),
            cameratimestamp TEXT,
            bottomPassFail TEXT,
            topPassFail TEXT,
            bottomAnomalyLevel REAL,
            topAnomalyLevel REAL,
            machineTimestamp TEXT,
            ActCycleTime INTEGER,
            ActClpClsTime INTEGER,
            ActCoolingTime INTEGER,
            ActCurrentServodrive_Disp_1 INTEGER,
            ActCurrentServodrive_Disp_2 INTEGER,
            ActCurrentServodrive_1 INTEGER,
            ActCurrentServodrive_2 INTEGER,
            ActCushionPosition INTEGER,
            ActFeedTemp INTEGER,
            ActFill INTEGER,
            ActFillTime_0 INTEGER,
            ActFillTime_1 INTEGER,
            ActFillTime_2 INTEGER,
            ActInjectionPos INTEGER,
            ActInjFillSpd INTEGER,
            ActCalEjtFwdSpd INTEGER,
            ActCalEjtRetSpd INTEGER,
            ActInjFwdStagePos_0 INTEGER,
            ActInjFwdStagePos_1 INTEGER,
            ActInjFwdStagePos_2 INTEGER,
            Inj_Act_Prs_0 INTEGER,
            Inj_Act_Prs_1 INTEGER,
            Inj_Act_Prs_2 INTEGER,
            ActInjFwdStagePrs_0 INTEGER,
            ActInjFwdStagePrs_1 INTEGER,
            ActInjFwdStagePrs_2 INTEGER,
            ActInjFwdStageTime_0 INTEGER,
            ActInjFwdStageTime_1 INTEGER,
            ActInjFwdStageTime_2 INTEGER,
            ActMotorRPMServodrive_0 INTEGER,
            ActMotorRPMServodrive_1 INTEGER,
            ActNozzleCurrent INTEGER,
            ActNozzlePIDPer INTEGER,
            ActNozzleTemp INTEGER,
            Actoiltemp INTEGER,
            ActProcOfMaxInjPrs INTEGER,
            ActProcOfMaxInjPrsPos INTEGER,
            ActRearSuckbackSpd INTEGER,
            ActRearSuckbackTime INTEGER,
            ActRefillTime INTEGER,
            ActSysPrsServodrive_0 INTEGER,
            ActSysPrsServodrive_1 INTEGER,
            ActTempServodrive_0 INTEGER,
            ActTempServodrive_1 INTEGER,
            ActTempServoMotor_0 INTEGER,
            ActTempServoMotor_1 INTEGER,
            ActCCprs INTEGER,
            ActZone1Temp INTEGER,
            ActZone2Temp INTEGER,
            ActZone3Temp INTEGER,
            ActZone4Temp INTEGER,
            ActZone5Temp INTEGER,
            ActZone6Temp INTEGER,
            PrvActInj1PlastTime INTEGER,
            Backprs_value INTEGER,
            ActProcMonMinInjPos INTEGER,
            flow_value INTEGER
        );
        """,
        """
        CREATE OR REPLACE FUNCTION match_documents (
            query_embedding VECTOR(768),
            match_count INT DEFAULT NULL,
            filter JSONB DEFAULT '{}'
        ) RETURNS TABLE (
            id BIGINT,
            content TEXT,
            metadata JSONB,
            similarity FLOAT
        ) LANGUAGE plpgsql AS $$
        BEGIN
            RETURN QUERY
            SELECT
                id,
                content,
                metadata,
                1 - (documents.embedding <=> query_embedding) AS similarity
            FROM documents
            WHERE metadata @> filter
            ORDER BY documents.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
        """
    ]

    for query in queries:
        try:
            supabase.rpc('execute_query', {'query': query}).execute()
        except Exception as e:
            print(f"Error executing query: {str(e)}")

def main():
    """Main function to start the file watcher."""
    logger.info("Calling main()\n")
    # WARN: do not uncomment unless this is a first time setup
    #setup_database()
    observer = Observer()
    event_handler = FileHandler()
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=True)
    observer.start()
    print(f"Watching directory: {WATCH_DIRECTORY}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
