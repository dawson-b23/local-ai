import logging
import pandas as pd
from database import insert_press20_data, insert_document_row
from models import Press20Data
import uuid
import asyncio
from langfuse import observe, get_client
import os
from dotenv import load_dotenv

load_dotenv()

langfuse = get_client()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@observe()
async def process_file(file_path: str):
    #langfuse.update_current_trace(metadata={"file_path": file_path})
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                data = Press20Data(**row.to_dict(), dataset_id=str(uuid.uuid4()))
                await insert_press20_data(data)
            #langfuse.update_current_trace(metadata={"status": "success", "file_type": "csv", "row_count": len(df)})
        else:
            dataset_id = str(uuid.uuid4())
            row_data = {"file_path": file_path, "processed_at": pd.Timestamp.now().isoformat()}
            await insert_document_row(dataset_id, row_data)
            #langfuse.update_current_trace(metadata={"status": "success", "file_type": file_path.split('.')[-1], "dataset_id": dataset_id})
        logger.info(f"Processed file: {file_path}")
        return "File processed successfully"
    except Exception as e:
        logger.error(f"** ** ** ERROR ** ** ** processing file: {str(e)}")
        #langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
        raise
