import streamlit as st
from streamlit_chat import message
import httpx
import asyncio
from models import QueryInput
from supabase import create_client, Client
from database import save_chat_message, get_chat_history, fetch_sessions
import uuid
import os
import re
from dotenv import load_dotenv
from langfuse import observe, get_client
from datetime import datetime
import logging

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# WARNING:
# change to False to get rid of auth (for testing)
# deployment requires this to be True or there is no tracing....
DEPLOY = False 

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:9999/rag-docs")

langfuse = get_client()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL", "http://localhost:8000")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="H&H AI Assistant", page_icon="🤖", layout="wide")

######################################
###
### log in 
###
######################################
#@observe()
async def sign_up(email: str, password: str):
    #langfuse.update_current_trace(user_id=email, metadata={"email": email})
    try:
        user = await asyncio.to_thread(supabase.auth.sign_up, {"email": email, "password": password})
        #langfuse.update_current_trace(metadata={"email": email, "status": "success"})
        return user
    except Exception as e:
        #langfuse.update_current_trace(metadata={"email": email, "status": "error", "error": str(e)})
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
        st.error(f"Logout failed: {str(e)}")

#@observe()
async def fetch_sessions():
    #langfuse.update_current_trace(metadata={"action": "fetch_sessions"})
    try:
        response = await asyncio.to_thread(supabase.table("chat_history").select("sessionid", distinct=True).execute)
        sessions = [row["sessionid"] for row in response.output]
        #langfuse.update_current_trace(metadata={"status": "success", "session_count": len(sessions)})
        return sessions
    except Exception as e:
        #langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
        return [st.session_state.session_id]

#@observe()
async def check_auth():
    #langfuse.update_current_trace(metadata={"action": "check_auth"})
    try:
        user = await asyncio.to_thread(supabase.auth.get_user)
        user_id = user.user.id if user and user.user else f"anon_{st.session_state.session_id}"
        user_email = user.user.email if user and user.user else "Anonymous"
        #langfuse.update_current_trace(metadata={"user_id": user_id, "status": "success"})
        return user_id, user_email
    except Exception as e:
        #langfuse.update_current_trace(metadata={"status": "error", "error": str(e)})
        return f"anon_{st.session_state.session_id}", "Anonymous"

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

######################################
###
### main app
###
######################################
@observe()
async def main_app():
    st.image("logo.png", width=150)
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = await get_chat_history(st.session_state.session_id)
    if "sessions" not in st.session_state:
        st.session_state.sessions = await fetch_sessions()
    if "user_id" not in st.session_state:
        st.session_state.user_id, st.session_state.user_email = await check_auth()

    with st.sidebar:
        st.header("H&H AI Assistant")
        st.image("logo.png", width=100)
        st.write(f"Logged in as: {st.session_state.user_email}")
        if st.button("Logout"):
            await sign_out()

        selected_session = st.selectbox("Select Session", st.session_state.sessions, key="session_select")
        if st.button("New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            if st.session_state.session_id not in st.session_state.sessions:
                st.session_state.sessions.append(st.session_state.session_id)
            st.rerun()

    st.title("H&H AI Assistant")
    st.write("Contact Dawson for issues: intern@hhmoldsinc.com | 832-977-3004")
    st.write("Docs: https://dawson-b23.github.io/HHDocs/")
    st.write("Try asking 'who are you', 'what can you do', or 'help' for quick info!")
    st.write("Ask your questions below:")

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if user_query := st.chat_input("Type your message here..."):
        if not user_query.strip():
            st.error("Please enter a valid query.")
            return
        user_message = {"role": "user", "content": user_query}
        st.session_state["messages"].append(user_message)
        await save_chat_message(st.session_state.user_id, st.session_state.session_id, user_message)
        st.chat_message("user").markdown(user_query)

        async with httpx.AsyncClient(timeout=float(os.getenv("HTTP_TIMEOUT", 30.0))) as client:
            with st.spinner("Processing..."):
                try:
                    langfuse.update_current_trace(session_id=st.session_state.session_id)
                    query_input = QueryInput(chatInput=user_query, sessionId=st.session_state.session_id)
                    response = await client.post(FASTAPI_URL, json=query_input.model_dump())
                    response.raise_for_status()
                    response_data = response.text
                    # Strip surrounding quotation marks
                    if response_data.startswith('"') and response_data.endswith('"'):
                        response_data = response_data[1:-1]
                        logger.debug(f"Stripped quotation marks from response: {response_data}")
                    # Validate response
                    if response_data.startswith("{") and response_data.endswith("}"):
                        logger.warning(f"Invalid response format from FastAPI: {response_data}")
                        response_data = "- Error: Invalid response format. Please try again or contact support."
                    elif not response_data.strip():
                        logger.warning("Empty response from FastAPI")
                        response_data = "- Error: No response received. Please try again or contact support."
                except httpx.TimeoutException:
                    logger.error("Timeout in FastAPI request", exc_info=True)
                    response_data = "- Error: Request timed out."
                except Exception as e:
                    logger.error(f"Error in FastAPI request: {str(e)}", exc_info=True)
                    response_data = f"- Error: {str(e)}"

                logger.debug(f"Final response to render: {response_data}")
                langfuse.update_current_trace(session_id=st.session_state.session_id)
                assistant_message = {"role": "assistant", "content": response_data}
                st.session_state["messages"].append(assistant_message)
                await save_chat_message(st.session_state.user_id, st.session_state.session_id, assistant_message)
                st.chat_message("assistant").markdown(response_data)

if __name__ == "__main__":
    if DEPLOY is True:
        if "user_id" not in st.session_state or "user_email" not in st.session_state:
            asyncio.run(auth_screen())
        else:
            asyncio.run(main_app())

    if DEPLOY is False:
        asyncio.run(main_app())
