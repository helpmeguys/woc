import streamlit as st
import json
import numpy as np
from openai import OpenAI
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from gtts import gTTS
from io import BytesIO
import time
import requests
import os
import urllib.parse
import pandas as pd
import altair as alt

# === CONFIGURATION ===
SITE_TITLE = st.secrets["SITE_TITLE"]
PASSWORD = st.secrets["ACCESS_PASSWORD"]
ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
REGISTRATION_URL = st.secrets["REGISTRATION_URL"]
DOWNLOAD_URL = st.secrets["DOWNLOAD_URL"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# === FILE PATHS ===
EMBEDDINGS_FILE = "qa_embeddings.json"
ACCESS_LOG_FILE = "access_log.json"
ACTIVITY_LOG_FILE = "activity_log.json"

# === PAGE CONFIG ===
st.set_page_config(page_title=f"📖 {SITE_TITLE}", layout="centered")

# === STYLING ===
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden; height: 0;}
        .block-container { padding-top: 1rem !important; }
        .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a {
            display: none;
        }
        .button-row {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .button-row button {
            padding: 0.5em 1em;
            border-radius: 6px;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
            cursor: pointer;
        }
        .button-row button:hover {
            background-color: #e0e0e0;
        }
    </style>
""", unsafe_allow_html=True)

# === FUNCTIONS ===
def log_access():
    now = datetime.now().strftime("%Y-%m")
    try:
        with open(ACCESS_LOG_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    data.append(now)
    with open(ACCESS_LOG_FILE, "w") as f:
        json.dump(data, f)

def get_monthly_usage():
    try:
        with open(ACCESS_LOG_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    return Counter(data)

def log_event(event_type, data=None):
    event = {
        "event": event_type,
        "timestamp": datetime.now().isoformat(),
    }
    if data:
        event.update(data)
    try:
        with open(ACTIVITY_LOG_FILE, "r") as f:
            logs = json.load(f)
    except FileNotFoundError:
        logs = []
    logs.append(event)
    with open(ACTIVITY_LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

client = OpenAI(api_key=OPENAI_API_KEY)

# === EMBEDDINGS ===
def ensure_embeddings_file():
    path = Path(EMBEDDINGS_FILE)
    if not path.exists():
        try:
            st.warning(f"🔄 Downloading {EMBEDDINGS_FILE} from {DOWNLOAD_URL}...")
            response = requests.get(DOWNLOAD_URL)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
            st.success("✅ Embeddings file downloaded.")
        except Exception as e:
            st.error(f"❌ Download failed: {e}")

ensure_embeddings_file()

@st.cache_data
def load_qa_embeddings():
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            item["embedding"] = np.array(item["embedding"])
        return data

qa_data = load_qa_embeddings()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@st.cache_data
def embed_query(text: str):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
    )
    return np.array(response.data[0].embedding)

def generate_tts_audio(text):
    try:
        tts = gTTS(text)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.warning(f"TTS failed: {e}")
        return None

# === UI ===
st.markdown(f"<h3 style='margin-bottom: 0.5rem;'>📖 {SITE_TITLE}</h3>", unsafe_allow_html=True)
st.markdown("Ask a question and receive an answers.")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown(f"<h3 style='margin-bottom: 0.5rem;'>🔐 {SITE_TITLE}</h3>", unsafe_allow_html=True)
    st.markdown(f"This tool is available <strong>free to registered users</strong>. Register here: [Click here to register]({REGISTRATION_URL})", unsafe_allow_html=True)
    password_input = st.text_input("Enter your password:", type="password")
    if password_input:
        if password_input == PASSWORD:
            st.session_state.authenticated = True
            log_access()
            st.success("Access granted. Welcome!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Incorrect password.")
            st.stop()
    else:
        st.stop()

query = st.text_input("Type your question or thought:")

if not query:
    col1, col2 = st.columns([2, 3])
    with col1:
        st.info("Enter a question to get started.")
    with col2:
        top_k = st.slider("", 1, 20, 5, help="Adjust how many insightful answers you'd like to see.")
else:
    top_k = st.slider("", 1, 20, 5, help="Adjust how many insightful answers you'd like to see.")
    log_event("search", {"query": query})

    with st.spinner("Searching by meaning..."):
        query_vec = embed_query(query)
        ranked_results = [(cosine_similarity(query_vec, item["embedding"]), item) for item in qa_data]
        ranked_results.sort(reverse=True, key=lambda x: x[0])
        top_results = ranked_results[:top_k]

        st.success(f"Top {top_k} matches:")
        for idx, (sim, qa) in enumerate(top_results):
            question = qa.get("question", "[No question]")
            answer = qa.get("answer", "[No answer]")
            section = qa.get("section_title", "[No section]")
            title = qa.get("video_title", "Untitled")
            timestamp = qa.get("timestamp", "0:00")
            url = qa.get("video_url", "#")

            st.markdown("----")
            st.markdown(f"**Q:** {question}")
            st.markdown(f"**A:** {answer}")
            if title.lower().strip() not in ["untitled", "untitled video", ""]:
                st.markdown(f"📖 *{section}* — **{title.strip()}**")
            else:
                st.markdown(f"📖 *{section}*")

            st.markdown(f"<a href='{url}' target='_blank'>▶️ Watch from {timestamp}</a>", unsafe_allow_html=True)

            mailto_body = f"Question: {question}\n\nAnswer: {answer}\n\nWatch here: {url}\n\nThese results were rendered at https://search.thewordsofchrist.org"
            mailto = f"mailto:?subject={urllib.parse.quote(SITE_TITLE)}&body={urllib.parse.quote(mailto_body)}"

            st.markdown(f"""
                <div class='button-row'>
                    <button onclick=\"navigator.clipboard.writeText('{url}');\">📋 Copy</button>
                    <a href='{mailto}' target='_blank'><button>📧 Email</button></a>
                    <button onclick=\"alert('🎧 Playing audio...');\">🎧 Listen</button>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"🔍 <span style='color:green;'>Semantic similarity: {sim:.3f}</span>", unsafe_allow_html=True)

# === LOGOUT BUTTON ===
st.markdown("---")
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button("🚪 Logout"):
    for key in list(st.session_state.keys()):
        if key.startswith("listen_") or key.startswith("audio_"):
            del st.session_state[key]
    st.session_state.authenticated = False
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# === FOOTER ===
usage = get_monthly_usage()
current_month = datetime.now().strftime("%Y-%m")
st.markdown(f"📊 **Logins this month:** `{usage.get(current_month, 0)}`")


