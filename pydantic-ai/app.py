# app.py (Streamlit UI - Enhanced with better session management, markdown rendering, help section)
import streamlit as st
from streamlit_chat import message
import httpx
import asyncio
import uuid
import os
from dotenv import load_dotenv
from supabase import create_client
from database import save_chat_message, get_chat_history, fetch_sessions
from models import QueryInput
from langfuse import observe, get_client
import logging
from utils.markdown import render_markdown
from agents.master_agent import master_agent, Deps

load_dotenv()

DEPLOY = True # Toggle auth

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:9999/query")
SUPABASE_URL = os.getenv("SUPABASE_URL", "http://localhost:8000")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
langfuse = get_client()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="H&H AI Assistant", page_icon="🤖", layout="wide")

async def sign_up(email: str, password: str):
    try:
        return await asyncio.to_thread(supabase.auth.sign_up, {"email": email, "password": password})
    except Exception as e:
        st.error(f"Sign-up failed: {str(e)}")
        return None

async def sign_in(email: str, password: str):
    try:
        return await asyncio.to_thread(supabase.auth.sign_in_with_password, {"email": email, "password": password})
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return None

async def sign_out():
    try:
        await asyncio.to_thread(supabase.auth.sign_out)
        st.session_state.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Logout failed: {str(e)}")

async def auth_screen():
    st.image("logo.png", width=150)
    st.title("🔐 H&H AI Assistant Login")
    option = st.selectbox("Action:", ["Login", "Sign Up"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if option == "Sign Up" and st.button("Register"):
        user = await sign_up(email, password)
        if user:
            st.success("Registered! Log in.")
    if option == "Login" and st.button("Login"):
        user = await sign_in(email, password)
        if user:
            st.session_state.user_id = user.user.id
            st.session_state.user_email = user.user.email
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.sessions = await fetch_sessions()  # Load sessions on login
            st.rerun()

@observe()
async def main_app():
    if "sessions" not in st.session_state:
        st.session_state.sessions = await fetch_sessions(st.session_state.user_id)
    if "session_id" not in st.session_state:
        st.session_state.session_id = st.session_state.sessions[0]['sessionid'] if st.session_state.sessions else str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = await get_chat_history(st.session_state.session_id)
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"anon_{st.session_state.session_id}"
        st.session_state.user_email = "Anonymous"

    with st.sidebar:
        st.header("H&H AI Assistant")
        st.image("logo.png", width=100)
        st.write(f"Logged in as: {st.session_state.user_email}")
        if DEPLOY and st.button("Logout"):
            await sign_out()
        
        # Session selector
        options = [f"{s['title']} (ID: {s['sessionid']}) - {s['timestamp']}" for s in st.session_state.sessions]
        current_index = next((i for i, opt in enumerate(options) if st.session_state.session_id in opt), 0)
        selected_option = st.selectbox("Select Conversation", options, index=current_index)
        selected_index = options.index(selected_option)
        selected_session = st.session_state.sessions[selected_index]['sessionid']
        if selected_session != st.session_state.session_id:
            st.session_state.session_id = selected_session
            st.session_state.messages = await get_chat_history(selected_session)
            st.rerun()
        
        if st.button("New Chat"):
            new_id = str(uuid.uuid4())
            await create_session(new_id, st.session_state.user_id)
            st.session_state.sessions = await fetch_sessions(st.session_state.user_id)
            st.session_state.session_id = new_id
            st.session_state.messages = []
            st.rerun()
        
        # Rename current chat
        current_session = next((s for s in st.session_state.sessions if s['sessionid'] == st.session_state.session_id), None)
        current_title = current_session['title'] if current_session else "New Chat"
        new_title = st.text_input("Rename current chat", value=current_title)
        if st.button("Rename") and new_title != current_title:
            await update_session_title(st.session_state.session_id, new_title)
            st.session_state.sessions = await fetch_sessions(st.session_state.user_id)
            st.rerun()
        
        st.markdown("**Help:** Ask about docs, press20 data, calculations, trends, defects.")
        st.markdown("[Docs](https://dawson-b23.github.io/HHDocs/)")
        st.markdown("Contact: intern@hhmoldsinc.com | 832-977-3004")

    st.title("H&H AI Assistant")
    st.markdown("Assist with injection molding: queries on Press20, docs, calculations, trends, defect fixes.")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(render_markdown(msg["content"]), unsafe_allow_html=True)

    if user_query := st.chat_input("Your question..."):
        if not user_query.strip():
            st.error("Enter valid query.")
            return
        user_msg = {"role": "user", "content": user_query}
        st.session_state.messages.append(user_msg)
        await save_chat_message(st.session_state.user_id, st.session_state.session_id, user_msg)
        with st.chat_message("user"):
            st.markdown(user_query)

        # Auto-update title if "New Chat"
        current_session = next((s for s in st.session_state.sessions if s['sessionid'] == st.session_state.session_id), None)
        if current_session and current_session['title'] == "New Chat":
            new_title = user_query[:50] + "..." if len(user_query) > 50 else user_query
            await update_session_title(st.session_state.session_id, new_title)
            st.session_state.sessions = await fetch_sessions(st.session_state.user_id)

        with st.spinner("Processing..."):
            try:
                # Construct full conversation history
                history_str = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]])
                full_input = f"{history_str}\n\nuser: {user_query}" if history_str else user_query

                async with httpx.AsyncClient(timeout=60.0) as client:
                    deps = Deps(client=client, supabase=supabase)
                    result = await master_agent.run(full_input, deps=deps)
                    logger.debug(f"Agent result: {result}")
                    response_text = result.output if hasattr(result, 'output') else str(result) if result else "- **No response.**"
                    rendered = render_markdown(response_text)
                    assistant_msg = {"role": "assistant", "content": rendered}
                    st.session_state.messages.append(assistant_msg)
                    await save_chat_message(st.session_state.user_id, st.session_state.session_id, assistant_msg)
                    with st.chat_message("assistant"):
                        st.markdown(rendered, unsafe_allow_html=True)
            except Exception as e:
                logger.error(f"Error in processing: {str(e)}", exc_info=True)
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    if DEPLOY and ("user_id" not in st.session_state):
        asyncio.run(auth_screen())
    else:
        asyncio.run(main_app())
