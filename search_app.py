# search_app.py (FAISS + Remote Download)

import streamlit as st
import json
import numpy as np
import faiss
from openai import OpenAI
from pathlib import Path
from datetime import datetime
from collections import Counter
from gtts import gTTS
from io import BytesIO
import time
import os
import requests
import urllib.request
import streamlit.components.v1 as components

# === CONFIGURATION ===
SITE_TITLE = os.environ.get("SITE_TITLE")
PASSWORD = os.environ.get("ACCESS_PASSWORD")
REGISTRATION_URL = os.environ.get("REGISTRATION_URL")
INDEX_FILE = "embeddings.index"
METADATA_FILE = "metadata.json"
INDEX_URL = os.environ.get("INDEX_URL")
META_URL = os.environ.get("META_URL")
ACCESS_LOG_FILE = "access_log.json"

# === PAGE CONFIG ===
st.set_page_config(page_title=f"📖 {SITE_TITLE}", layout="centered")

# === STYLING ===
st.markdown("""
    <style>
        .stDeployButton, [data-testid="stStatusWidget"], .viewerBadge_container__1QSob,
        .stActionButtonIcon, div[class*="floating"], [data-testid="collapsedControl"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# === DOWNLOAD IF MISSING ===
def download_if_missing(file_path, url):
    if not Path(file_path).exists():
        alert_box = st.empty()
        alert_box.warning(f"📅 Downloading {file_path} from remote...")
        time.sleep(2)
        alert_box.empty()  # clears the message

        urllib.request.urlretrieve(url, file_path)

        alert_box = st.empty()
        alert_box.success(f"✅ Downloaded {file_path}")
        time.sleep(2)
        alert_box.empty()


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

# === TEXT-TO-SPEECH ===
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
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# === DOWNLOAD + LOAD INDEX ===
download_if_missing(INDEX_FILE, INDEX_URL)
download_if_missing(METADATA_FILE, META_URL)

@st.cache_resource
def load_faiss_index():
    return faiss.read_index(INDEX_FILE)

@st.cache_data
def load_metadata():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

index = load_faiss_index()
metadata = load_metadata()

# === EMBEDDING & SEARCH ===
def embed_query(text: str):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
    )
    return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

# === YOUTUBE HELPERS ===
def extract_youtube_video_id(url):
    """Extract YouTube video ID from a URL"""
    if not url or "youtube.com" not in url and "youtu.be" not in url:
        return None
        
    try:
        if "youtube.com/watch" in url:
            # Handle youtube.com/watch?v=VIDEO_ID format
            from urllib.parse import urlparse, parse_qs
            parsed_url = urlparse(url)
            return parse_qs(parsed_url.query)["v"][0]
        elif "youtu.be/" in url:
            # Handle youtu.be/VIDEO_ID format
            return url.split("youtu.be/")[1].split("?")[0]
        else:
            return None
    except Exception:
        return None
        
def get_youtube_thumbnail_url(video_id):
    """Get the thumbnail URL for a YouTube video"""
    if not video_id:
        return ""
    # Use the maxresdefault image when available (highest quality)
    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    
def get_youtube_embed_html(video_id, timestamp=None):
    """Generate HTML to embed a YouTube video with optional timestamp"""
    if not video_id:
        return ""
        
    # Convert timestamp format (e.g., "1:23:45" or "1:23") to seconds for YouTube embed
    start_seconds = 0
    if timestamp:
        parts = timestamp.split(":")
        if len(parts) == 3:  # hours:minutes:seconds
            start_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:  # minutes:seconds
            start_seconds = int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 1 and parts[0].isdigit():  # seconds
            start_seconds = int(parts[0])
            
    # Create an embedded YouTube player with autoplay disabled and modest branding
    # The ?start parameter specifies where to start the video in seconds
    embed_url = f"https://www.youtube.com/embed/{video_id}?start={start_seconds}&rel=0&modestbranding=1"
    
    return f"""
    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; border-radius: 8px;">
        <iframe 
            src="{embed_url}" 
            style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0;" 
            allowfullscreen 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture">
        </iframe>
    </div>
    """

def search_faiss(query_vector, top_k):
    # Request more results than needed to account for potential duplicates
    expanded_k = min(top_k * 3, len(metadata))  # Request 3x but don't exceed total data size
    scores, indices = index.search(query_vector, expanded_k)
    
    # Get actual number of results returned (may be less than expanded_k)
    actual_results = min(expanded_k, len(indices[0]))
    
    # Track unique video+timestamp combinations and keep only the highest scoring result for each
    unique_results = {}
    for i in range(actual_results):
        idx = indices[0][i]
        if idx >= 0 and idx < len(metadata):
            meta = metadata[idx]
            # Create a unique key based on video URL and timestamp
            video_url = meta.get("video_url", "")
            timestamp = meta.get("timestamp", "")
            unique_key = f"{video_url}_{timestamp}"
            
            # Only keep the highest-scoring result for each unique key
            if unique_key not in unique_results or scores[0][i] > unique_results[unique_key][0]:
                unique_results[unique_key] = (scores[0][i], meta)
    
    # Convert the dictionary back to a list and sort by score (descending)
    results = list(unique_results.values())
    results.sort(key=lambda x: x[0], reverse=True)
    
    # Limit to the originally requested number of results
    return results[:top_k]

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

    with st.spinner("🔍 Searching by meaning..."):
        query_vec = embed_query(query)
        top_results = search_faiss(query_vec, top_k)

        if not top_results:
            st.warning("⚠️ No results found for your query. Please try a different question.")
        else:
            st.success(f"Found {len(top_results)} matches for your question:")
            for idx, (sim, qa) in enumerate(top_results):
                try:
                    question = qa.get("question", "[No question]")
                    answer = qa.get("answer", "[No answer]")
                    title = qa.get("video_title", "untitled")
                    timestamp = qa.get("timestamp", "0:00")
                    url = qa.get("video_url", "#")
                    segment = qa.get("segment_title", "")

                    st.markdown("----")
                    
                    # Extract video ID for embedding
                    video_id = extract_youtube_video_id(url)
                    
                    # Display embedded YouTube player if video ID is available
                    if video_id:
                        embed_html = get_youtube_embed_html(video_id, timestamp)
                        components.html(embed_html, height=350)
                    
                    if title.lower().strip() not in ["untitled", "untitled video", ""]:
                        st.markdown(f"📖 **{title}**")
                    else:
                        st.markdown(f"📖 **Video**")
                        
                    # Display segment title if available
                    if segment and segment.lower().strip() not in ["", "untitled"]:
                        st.markdown(f"📝 **Segment:** {segment}")
                        
                    st.markdown(f"⏰ **Timestamp:** {timestamp}")
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        components.html(f"""
                        <div>
                            <button onclick="navigator.clipboard.writeText('{url}'); this.innerText='✅ Copied!'; setTimeout(() => this.innerText='📋 Copy link', 2000);" style="cursor:pointer; padding:4px 10px; font-size:0.85rem; border:1px solid #ccc; border-radius:5px; background:#f9f9f9;">📋 Copy link</button>
                        </div>
                        """, height=40)
                    with col2:
                        components.html(f"""
                        <div>
                            <a href="{url}" target="_blank" style="display:inline-block; padding:4px 10px; font-size:0.85rem; border:1px solid #FF0000; border-radius:5px; background:#f9f9f9; color:#FF0000; text-decoration:none;">📺 Open in YouTube</a>
                        </div>
                        """, height=40)

                    st.markdown(
                        f" **<span style='color:green;'>Semantic similarity: {sim:.3f}</span>**",
                        unsafe_allow_html=True
                    )

                    with st.expander("🎰 Listen to this answer"):
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
st.markdown("[💡 Powered by AskClips.com](https://askclips.com)", unsafe_allow_html=True)
