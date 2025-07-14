from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import httpx
from database import setup_vector_store
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import json
import asyncio
from langfuse import get_client, observe

load_dotenv()


client = AsyncOpenAI(
    base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
    api_key="ollama"
)
#model = OpenAIModel("llama3.1:8b", openai_client=client)
model = OpenAIModel(
    model_name='llama3.1:8b', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

langfuse = get_client()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL", "http://localhost:8000")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@dataclass
class Deps:
    client: httpx.AsyncClient
    supabase_key: str
    supabase: 'supabase.Client'

rag_agent = Agent(
    model,
    system_prompt="""
    You are a personal assistant who helps answer questions from a corpus of documents. The documents are either 
    text based (Txt, docs, extracted PDFs, etc.) or tabular data (CSVs or Excel documents).

    You are given tools to perform RAG in the 'documents' table, look up the documents 
    available in your knowledge base in the 'document_metadata' table, extract all the text 
    from a given document, and query the tabular files with SQL in the 'document_rows' table. 

    Always start by performing RAG unless the question requires a SQL query for tabular data 
    (fetching a sum, finding a max, something a RAG lookup would be unreliable for). 
    If RAG doesn't help, then look at the documents that are available to you, find a few 
    that you think would contain the answer, and then analyze those.

    Always tell the user if you didn't find the answer. Don't make something up just to please them. 
    If you decide to return an sql query, try to run it first and return the result.

    YOU MUST RETURN FINAL ANSWERS IN MARKDOWN FORMAT.
    """,
#    system_prompt="""You are a document query assistant for H&H. Use the provided tools to retrieve information from documents or tabular data. Follow these steps:
#1. For text-based queries (e.g., 'summarize', 'what is'), use list documents to see if any of them match, then rag_search (vector store) first to retrieve relevant documents.
#2. For queries needing document metadata, use list_documents.
#3. For specific file text, use get_file_contents with a file_id.
#4. For tabular data (e.g., sums, averages), use query_document_rows with a SQL query.
#5. If no relevant data is found, return 'No relevant data found.
#6. For summarization queries, use rag_search to retrieve documents, then summarize the content using the model.""",
    #"""You are a document query assistant for H&H. Use the provided tools to search and retrieve information from documents or tabular data. Start with a RAG search for text-based queries. For tabular data queries (e.g., sums, averages), use the Query Document Rows tool. Use List Documents to check available documents and Get File Contents to extract text for a specific file_id. If no relevant data is found, return 'No relevant data found.'""",
    #system_prompt="""
    #You are a document query assistant for H&H. Search and retrieve relevant information 
    #from documents or chat history based on user input. Use the vector store for document 
    #searches and query the documents, document_rows, or chat_history tables as needed. 
    #Return the retrieved content or an error message if no relevant data is found. 
    #""",
    #system_prompt="""You are a document query assistant for H&H. Search and retrieve relevant 
    #information from documents using the vector store. If no relevant data is found, return 
    #'No relevant documents found.'""",
    deps_type=Deps,
    retries=3
)

@rag_agent.tool
async def rag_search(ctx: RunContext[Deps], query: str) -> str:
    try:
        vector_store = await setup_vector_store()
        results = await vector_store.as_retriever().ainvoke(query)
        if not results:
            return "No relevant documents found."
        # For summarization queries, generate a summary
        #if "summarize" in query.lower():
        #    content = "\n".join([str(doc.page_content) for doc in results])
        #    summary_prompt = f"Summarize the following content concisely:\n{content}"
        #    response = await client.chat.completions.create(
        #        model="llama3.1:8b",
        #        messages=[
        #            {"role": "system", "content": "You are a summarization assistant. Provide a concise summary of the provided content."},
        #            {"role": "user", "content": summary_prompt}
        #        ]
        #    )
        #    return response.choices[0].message.content
        return "\n".join([str(doc.page_content) for doc in results])
    except Exception as e:
        return f"Error in RAG search: {str(e)}"
#async def rag_search(ctx: RunContext[Deps], query: str) -> str:
#    try:
#        vector_store = await setup_vector_store()
#        results = await vector_store.as_retriever().ainvoke(query)
#        if not results:
#            return "No relevant documents found."
#        return "\n".join([str(doc.page_content) for doc in results])
#    except Exception as e:
#        return f"Error searching documents: {str(e)}"


@rag_agent.tool
@observe()
async def list_documents(ctx: RunContext[Deps]) -> str:
    try:
        response = await asyncio.to_thread(
            supabase.table("document_metadata").select("*").execute
        )
        if not response.output:
            return "No documents found in document_metadata."
        return json.dumps(response.output, indent=2)
    except Exception as e:
        return f"Error listing documents: {str(e)}"


@rag_agent.tool
@observe()
async def get_file_contents(ctx: RunContext[Deps], file_id: str) -> str:
    try:
        response = await asyncio.to_thread(
            supabase.rpc(
                "get_file_contents",
                {"file_id": file_id}
            ).execute
        )
        if not response.output:
            return f"No text found for file_id: {file_id}"
        return response.output[0].get("document_text", "No text content")
    except Exception as e:
        return f"Error getting file contents: {str(e)}"


@rag_agent.tool
@observe()
async def query_document_rows(ctx: RunContext[Deps], sql_query: str) -> str:
    try:
        response = await asyncio.to_thread(
            supabase.rpc(
                "query_document_rows",
                {"query_text": sql_query}
            ).execute
        )
        if not response.output:
            return "No data found in document_rows."
        return json.dumps(response.output, indent=2)
    except Exception as e:
        return f"Error querying document rows: {str(e)}"
