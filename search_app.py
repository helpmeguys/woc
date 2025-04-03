# gospel_search_app.py

import streamlit as st
import json
import numpy as np
from openai import OpenAI
from pathlib import Path
from datetime import datetime
from collections import Counter
from gtts import gTTS
from io import BytesIO
import time
import requests
import os
import streamlit.components.v1 as components

# === PRE-INIT CONFIG (must be BEFORE page config!)
SITE_TITLE = st.secrets.get("SITE_TITLE", "Gospel Q&A")
PASSWORD = st.secrets["ACCESS_PASSWORD"]
REGISTRATION_URL = st.secrets["REGISTRATION_URL"]
DOWNLOAD_URL = st.secrets["DOWNLOAD_URL"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# === PAGE CONFIG ===
st.set_page_config(page_title=f"📖 {SITE_TITLE}", layout="centered")

# === STYLE FIXES FOR HEADER/LINK ICONS ===
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden; height: 0;}
        .block-container { padding-top: 1rem !important; }
        h1, h2, h3, h4, h5, h6 {
            visibility: visible;
            margin-top: 0 !important;
        }
        .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# === INIT SESSION STATE FOR LOGIN ===
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# === LOGIN TRACKING ===
ACCESS_LOG_FILE = "access_log.json"
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

# === AUTHENTICATION ===
if not st.session_state.authenticated:
    st.markdown(f"<h3 style='margin-bottom: 0.5rem;'>🔐 {SITE_TITLE}</h3>", unsafe_allow_html=True)
    st.markdown(f"This tool is available <strong>free to registered users</strong>. Register here: [Click here to register]({REGISTRATION_URL})", unsafe_allow_html=True)
    password = st.text_input("Enter your password:", type="password")

    if password == PASSWORD:
        st.session_state.authenticated = True
        log_access()
        st.success("Access granted. Welcome!")
        time.sleep(1.5)
        st.experimental_rerun()
    elif password:
        st.error("Incorrect password.")
        st.stop()
    else:
        st.stop()

# === OPENAI SETUP ===
client = OpenAI(api_key=OPENAI_API_KEY)

# === EMBEDDINGS ===
EMBEDDINGS_FILE = "qa_embeddings.json"

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

def embed_query(text: str):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
    )
    return np.array(response.data[0].embedding)

# === TEXT-TO-SPEECH ===
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

# === MAIN UI ===
st.markdown(f"<h3 style='margin-bottom: 0.5rem;'>📖 {SITE_TITLE}</h3>", unsafe_allow_html=True)
st.markdown("Ask a gospel question and receive the most relevant spiritually insightful answers, matched by meaning.")

query = st.text_input("Type your gospel question or thought:")
top_k = st.sidebar.slider("🔧 Number of Results", 1, 20, 5)

if query:
    with st.spinner("Searching by meaning..."):
        query_vec = embed_query(query)
        ranked_results = [(cosine_similarity(query_vec, item["embedding"]), item) for item in qa_data]
        ranked_results.sort(reverse=True, key=lambda x: x[0])
        top_results = ranked_results[:top_k]

        st.success(f"Top {top_k} matches:")
        for sim, qa in top_results:
            question = qa.get("question", "[No question]")
            answer = qa.get("answer", "[No answer]")
            section = qa.get("section_title", "[No section]")
            title = qa.get("video_title", "Untitled")
            timestamp = qa.get("timestamp", "0:00")
            url = qa.get("video_url", "#")

            st.markdown("----")
            st.markdown(f"**Q:** {question}")
            st.markdown(f"**A:** {answer}")
            if title.lower().strip() not in ["untitled", ""]:
                st.markdown(f"📖 *{section}* — **{title}**")
            else:
                st.markdown(f"📖 *{section}*")
            st.markdown(f"<a href='{url}' target='_blank'>▶️ Watch from {timestamp}</a>", unsafe_allow_html=True)

            components.html(f"""
                <div style='margin-top:4px;'>
                    <button onclick="navigator.clipboard.writeText('{url}'); this.innerText='✅ Copied!'; setTimeout(() => this.innerText='📋 Copy link', 2000);" style="cursor:pointer; padding:4px 10px; font-size:0.85rem; border:1px solid #ccc; border-radius:5px; background:#f9f9f9;'>📋 Copy link</button>
                </div>
            """, height=40)

            st.markdown(f"🔍 <span style='color:green;'>Semantic similarity: {sim:.3f}</span>", unsafe_allow_html=True)

            with st.expander("🎧 Listen to this answer"):
                audio = generate_tts_audio(f"Question: {question}. Answer: {answer}")
                if audio:
                    st.audio(audio, format="audio/mp3")
else:
    st.info("Enter a gospel-related question to get started.")

# === FOOTER: USAGE TRACKING ===
usage = get_monthly_usage()
current_month = datetime.now().strftime("%Y-%m")
st.markdown("---")
st.markdown(f"📈 **Logins this month:** `{usage.get(current_month, 0)}`")

