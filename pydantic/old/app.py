import streamlit as st
from streamlit_chat import message
import httpx
import asyncio
from supabase import create_client, Client
from models import QueryInput
from database import save_chat_message, get_chat_history
from agents.master_agent import create_master_agent, Deps
import uuid
import os
from dotenv import load_dotenv
from langfuse import observe, get_client
import logging
import json

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
langfuse = get_client()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL", "http://localhost:8000")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Define FastAPI endpoint
FASTAPI_URL = "http://localhost:9999/rag-docs"

st.set_page_config(page_title="H&H AI Assistant", page_icon="🤖", layout="wide")



#@observe()
async def sign_up(email: str, password: str):
    #langfuse.update_current_trace(user_id=email, metadata={"email": email})
    try:
        user = await asyncio.to_thread(supabase.auth.sign_up, {"email": email, "password": password})
        #langfuse.update_current_trace(metadata={"email": email, "status": "success"})
        return user
    except Exception as e:
        #langfuse.update_current_trace(metadata={"email": email, "status": "error", "error": str(e)})
        logger.error(f"** ** ** ERROR ** ** ** in sign up: {str(e)}", exc_info=True)
        st.error(f"Registration failed: {str(e)}")
        return None

#@observe()
async def sign_in(email: str, password: str):
    #langfuse.update_current_trace(user_id=email, metadata={"email": email})
    try:
        user = await asyncio.to_thread(supabase.auth.sign_in_with_password, {"email": email, "password": password})
        #langfuse.update_current_trace(metadata={"email": email, "status": "success"})
        return user
    except Exception as e:
        #langfuse.update_current_trace(metadata={"email": email, "status": "error", "error": str(e)})
        logger.error(f"** ** ** ERROR ** ** ** in sign in: {str(e)}", exc_info=True)
        st.error(f"Login failed: {str(e)}")
        return None

#@observe()
async def sign_out():
    #langfuse.update_current_trace(metadata={"action": "sign_out"})
    try:
        await asyncio.to_thread(supabase.auth.sign_out)
        st.session_state.user_id = None
        st.session_state.user_email = None
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.sessions = []
        #langfuse.update_current_trace(metadata={"status": "success"})
        st.rerun()
    except Exception as e:
        #langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
        logger.error(f"** ** ** ERROR ** ** ** in sign out: {str(e)}", exc_info=True)
        st.error(f"Logout failed: {str(e)}")

@observe()
async def fetch_sessions():
    #langfuse.update_current_trace(metadata={"action": "fetch_sessions"})
    try:
        response = await asyncio.to_thread(supabase.table("chat_history").select("sessionid", distinct=True).execute)
        sessions = [row["sessionid"] for row in response.data]
        #langfuse.update_current_trace(metadata={"status": "success", "session_count": len(sessions)})
        return sessions
    except Exception as e:
        #langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
        logger.error(f"** ** ** ERROR ** ** ** in fetch_sessions: {str(e)}", exc_info=True)
        return [st.session_state.session_id]

@observe()
async def check_auth():
    #langfuse.update_current_trace(metadata={"action": "check_auth"})
    try:
        user = await asyncio.to_thread(supabase.auth.get_user)
        user_id = user.user.id if user and user.user else f"anon_{st.session_state.session_id}"
        user_email = user.user.email if user and user.user else "Anonymous"
        #langfuse.update_current_trace(metadata={"user_id": user_id, "status": "success"})
        return user_id, user_email
    except Exception as e:
        logger.error(f"** ** ** ERROR ** ** ** in check auth: {str(e)}", exc_info=True)
        #langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
        return f"anon_{st.session_state.session_id}", "Anonymous"

@observe()
async def ingest_file(file_path: str):
    #langfuse.update_current_trace(metadata={"file_path": file_path})
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{FASTAPI_URL.replace('/rag-docs', '')}/ingest-file", json={"file_path": file_path})
            response.raise_for_status()
            #langfuse.update_current_trace(metadata={"status": "success"})
            return response.json().get("status", "File ingested successfully")
        except Exception as e:
            #langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
            logger.error(f"** ** ** ERROR ** ** ** in ingest files: {str(e)}", exc_info=True)
            raise

@observe()
async def main_app():
    st.image("logo.png", width=150)  # Company logo

    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = await get_chat_history(st.session_state.session_id)
    if "sessions" not in st.session_state:
        st.session_state.sessions = await fetch_sessions()
    if "user_id" not in st.session_state:
        st.session_state.user_id, st.session_state.user_email = await check_auth()

    # Sidebar
    with st.sidebar:
        st.header("H&H AI Assistant")
        st.image("logo.png", width=100)  # Smaller logo in sidebar
        st.write(f"Logged in as: {st.session_state.user_email}")
        if st.button("Logout"):
            await sign_out()
        
        st.header("Chat Sessions")
        selected_session = st.selectbox("Select Session", st.session_state.sessions, key="session_select")
        if st.button("New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            if st.session_state.session_id not in st.session_state.sessions:
                st.session_state.sessions.append(st.session_state.session_id)
        if st.button("Delete Session"):
            #langfuse.update_current_trace(metadata={"action": "delete_session", "session_id": st.session_state.session_id})
            try:
                await asyncio.to_thread(supabase.table("chat_history").delete().eq("sessionid", st.session_state.session_id).execute)
                st.session_state.sessions.remove(st.session_state.session_id)
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                #langfuse.update_current_trace(metadata={"status": "success"})
            except Exception as e:
                #langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
                st.error(f"** ** ** ERROR ** ** ** deleting session: {str(e)}")

        st.header("File Ingestion")
        uploaded_file = st.file_uploader("Upload a file (PDF, CSV, Excel, DOCX)", type=["pdf", "csv", "xlsx", "docx"])
        if uploaded_file:
            file_path = f"/tmp/{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            result = await ingest_file(file_path)
            st.success(result)

    # Main chat interface
    st.title("H&H AI Assistant")
    st.write("Please let Dawson know if you have any strange behavior or bugs:")
    st.write("Email: intern@hhmoldsinc.com || Phone: (832-977-3004)")
    st.write("Usage: See docs @ https://dawson-b23.github.io/HHDocs/")

    # Display conversation history
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # Input box for user query at the bottom
    if user_query := st.chat_input("Type your message here..."):
        # Add user message to session state
        st.session_state["messages"].append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        # Agent response
        agent = create_master_agent()
        with st.spinner("Agent is thinking..."):
            try:
                # Run the agent and get the result
                result = asyncio.run(agent.run(user_query))
                
                # Extract the 'content' field from the JSON response
                response_content = result.data  # Assuming result.data is parsed into FreeFormResponse

                # Add agent response to session state
                st.session_state["messages"].append({"role": "assistant", "content": response_content})
                st.chat_message("assistant").write(response_content)

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.session_state["messages"].append({"role": "assistant", "content": error_message})
                st.chat_message("assistant").write(error_message)

async def auth_screen():
    st.image("logo.png", width=150)  # Company logo
    st.title("🔐 H&H AI Assistant Login")
    option = st.selectbox("Choose an action:", ["Login", "Sign Up"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if option == "Sign Up" and st.button("Register"):
        user = await sign_up(email, password)
        if user and user.user:
            st.success("Registration successful. Please log in.")

    if option == "Login" and st.button("Login"):
        user = await sign_in(email, password)
        if user and user.user:
            st.session_state.user_id = user.user.id
            st.session_state.user_email = user.user.email
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.sessions = []
            st.success(f"Welcome back, {email}!")
            st.rerun()

if __name__ == "__main__":
    if "user_id" not in st.session_state or "user_email" not in st.session_state:
        asyncio.run(auth_screen())
    else:
        asyncio.run(main_app())
