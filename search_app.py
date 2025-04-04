# search_app.py

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

# === CONFIGURATION ===
SITE_TITLE = os.environ.get("SITE_TITLE")
PASSWORD = os.environ.get("ACCESS_PASSWORD")
REGISTRATION_URL = os.environ.get("REGISTRATION_URL")
DOWNLOAD_URL = os.environ.get("DOWNLOAD_URL")

# === FILE PATHS ===
EMBEDDINGS_FILE = "qa_embeddings.json"
ACCESS_LOG_FILE = "access_log.json"


# === PAGE CONFIG ===
st.set_page_config(page_title=f"📖 {SITE_TITLE}", layout="centered")

# === STYLING ===
st.markdown("""
    <style>
        /* Hide Streamlit deploy button */
        .stDeployButton { display: none !important; }

        /* Hide status widget (crown icon) */
        [data-testid="stStatusWidget"] { display: none !important; }

        /* Hide floating chat/help button */
        .viewerBadge_container__1QSob { display: none !important; }
        .stActionButtonIcon { display: none !important; }

        /* Try hiding anything with class that includes 'floating' */
        div[class*="floating"] { display: none !important; }

        /* Absolute override if previous fails */
        [data-testid="collapsedControl"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)


# === SESSION STATE INIT ===
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# === LOGIN TRACKING ===
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

# === TEXT-TO-SPEECH GENERATOR ===
def generate_tts_audio(text):
    try:
        tts = gTTS(text)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.warning(f"⚠️ TTS generation failed: {e}")
        return None

# === PASSWORD GATE ===
if not st.session_state.authenticated:
    st.markdown(f"<h3 style='margin-bottom: 0.5rem;'>🔐 {SITE_TITLE}</h3>", unsafe_allow_html=True)
    st.markdown(f"This tool is available <strong>free to registered users</strong>. Register here: [Click here to register]({REGISTRATION_URL})", unsafe_allow_html=True)
    password = st.text_input("Enter your password:", type="password")
    if password == PASSWORD:
        st.session_state.authenticated = True
        log_access()
        st.success("Access granted! Welcome.")
        time.sleep(2)
        st.rerun()
    elif password:
        st.error("Incorrect password.")
        st.stop()
    else:
        st.stop()

# === OPENAI SETUP ===
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# === Ensure Embeddings File Exists ===
def ensure_embeddings_file():
    path = Path(EMBEDDINGS_FILE)
    if not path.exists():
        try:
            st.warning(f"🔄 Downloading {EMBEDDINGS_FILE} from {DOWNLOAD_URL}...")
            response = requests.get(DOWNLOAD_URL)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
            st.success("✅ Embeddings file downloaded successfully.")
        except Exception as e:
            st.error(f"❌ Could not download the embeddings file: {e}")

ensure_embeddings_file()

# === LOAD EMBEDDED DATA ===
@st.cache_data
def load_qa_embeddings():
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            item["embedding"] = np.array(item["embedding"])
        return data

qa_data = load_qa_embeddings()

# === SEMANTIC SEARCH ===
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def embed_query(text: str):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
    )
    return np.array(response.data[0].embedding)

# === MAIN UI ===
st.markdown(f"<h3 style='margin-bottom: 0.5rem;'>🔎 {SITE_TITLE}</h3>", unsafe_allow_html=True)
st.markdown("Ask a question and receive an answers.")

query = st.text_input("Type your question or thought:")

if not query:
    col1, col2 = st.columns([2, 3])
    with col1:
        st.info("Enter a question to get started.")
    with col2:
        top_k = st.slider(
            label="",
            min_value=1,
            max_value=20,
            value=5,
            help="Adjust how many insightful answers you'd like to see."
        )
else:
    with st.container():
        top_k = st.slider(
            label="",
            min_value=1,
            max_value=20,
            value=5,
            help="Adjust how many insightful answers you'd like to see."
        )
        
    with st.spinner("Searching by meaning..."):
        query_vec = embed_query(query)
        ranked_results = []

        for item in qa_data:
            sim = cosine_similarity(query_vec, item["embedding"])
            ranked_results.append((sim, item))

        ranked_results.sort(reverse=True, key=lambda x: x[0])
        top_results = ranked_results[:top_k]

        st.success(f"Top {top_k} matches for your question:")
        for idx, (sim, qa) in enumerate(top_results):
            try:
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
                    st.markdown(f"📖 *{section}* — **{title}**")
                else:
                    st.markdown(f"📖 *{section}*")

                st.markdown(f"<a href='{url}' target='_blank'>▶️ Watch from {timestamp}</a>", unsafe_allow_html=True)

                components.html(f"""
                <div style='margin-top:4px;'>
                    <button onclick="navigator.clipboard.writeText('{url}'); this.innerText='✅ Copied!'; setTimeout(() => this.innerText='📋 Copy link', 2000);" style="cursor:pointer; padding:4px 10px; font-size:0.85rem; border:1px solid #ccc; border-radius:5px; background:#f9f9f9;">📋 Copy link</button>
                </div>
                """, height=40)

                st.markdown(
                    f" **<span style='color:green;'>Semantic similarity: {sim:.3f}</span>**",
                    unsafe_allow_html=True
                )

                with st.expander("🎧 Listen to this answer"):
                    audio = generate_tts_audio(f"Question: {question}. Answer: {answer}")
                    if audio:
                        st.audio(audio, format="audio/mp3")
            except Exception as e:
                st.warning(f"⚠️ Error displaying result: {e}")
                
                
                
# === LOGOUT BUTTON ===
st.markdown("---")
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# === FOOTER ===
usage = get_monthly_usage()
current_month = datetime.now().strftime("%Y-%m")
st.markdown(f"📊 **Logins this month:** `{usage.get(current_month, 0)}`")
st.markdown("---")
st.markdown("[💡 Powered by YourTechDad.com](https://yourtechdad.com)", unsafe_allow_html=True)
