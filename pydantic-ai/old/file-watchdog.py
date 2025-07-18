import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ingestion import process_file
import logging
import asyncio
from langfuse import observe, get_client
import os
from dotenv import load_dotenv

load_dotenv()

langfuse = get_client()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FileHandler(FileSystemEventHandler):
    @observe()
    async def on_created(self, event):
        if not event.is_directory:
            #langfuse.update_current_trace(metadata={"file_path": event.src_path})
            logger.info(f"New file detected: {event.src_path}")
            try:
                await process_file(event.src_path)
                #langfuse.update_current_trace(metadata={"status": "success"})
            except Exception as e:
                #langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
                logger.error(f"** ** ** ERROR ** ** ** processing file: {str(e)}")

@observe()
async def watch_directory(directory: str):
    #langfuse.update_current_trace(metadata={"directory": directory})
    try:
        event_handler = FileHandler()
        observer = Observer()
        observer.schedule(event_handler, directory, recursive=False)
        observer.start()
        logger.info(f"Watching directory: {directory}")
        #langfuse.update_current_trace(metadata={"status": "started"})
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            langfuse.update_current_trace(metadata={"status": "stopped"})
        observer.join()
    except Exception as e:
        #langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
        logger.error(f"** ** ** ERROR ** ** ** in file watcher: {str(e)}")

if __name__ == "__main__":
    asyncio.run(watch_directory("/tank/local-ai/data"))
