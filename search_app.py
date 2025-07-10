# search_app.py (FAISS + Remote Download)

import streamlit as st
import json
import numpy as np
import faiss
from openai import OpenAI
from pathlib import Path
from datetime import datetime
from collections import Counter

import time
import os
import requests
import urllib.request
import streamlit.components.v1 as components
import streamlit_player as st_player
from pathlib import Path

# === CONFIGURATION ===
SITE_TITLE = os.environ.get("SITE_TITLE")
PASSWORD = os.environ.get("ACCESS_PASSWORD")
REGISTRATION_URL = os.environ.get("REGISTRATION_URL")
INDEX_FILE = "embeddings.index"
METADATA_FILE = "metadata.json"
INDEX_URL = os.environ.get("INDEX_URL")
META_URL = os.environ.get("META_URL")
LAST_UPDATED = os.environ.get("LAST_UPDATED")

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
ACCESS_LOG_FILE = DATA_DIR / "access_log.json"

# Customzation
PROFILE_PICTURE_URL = os.environ.get("PROFILE_PICTURE_URL")

# === PAGE CONFIG ===
st.set_page_config(page_title=SITE_TITLE, 
                 layout="centered",
                 page_icon="https://f000.backblazeb2.com/file/megatransfer/vaults/AC_Logo_Small.png")

# === STYLING ===
st.markdown("""
    <style>
        /* Hide Streamlit UI elements */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton, [data-testid="stStatusWidget"], .viewerBadge_container__1QSob,
        .stActionButtonIcon, div[class*="floating"], [data-testid="collapsedControl"] {
            display: none !important;
        }
        
        /* Remove top padding */
        .stApp {
            margin-top: -25px;
            padding-top: 0;
        }
        
        /* Adjust main container spacing */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
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
    data = []
    try:
        if ACCESS_LOG_FILE.exists():
            with open(ACCESS_LOG_FILE, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):  # Handle case where file exists but is not a list
                    data = []
    except (json.JSONDecodeError, FileNotFoundError):
        data = []
    
    data.append(now)
    
    # Ensure the directory exists before writing
    ACCESS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ACCESS_LOG_FILE, "w") as f:
        json.dump(data, f)

def get_monthly_usage():
    if not ACCESS_LOG_FILE.exists():
        return {}
    try:
        with open(ACCESS_LOG_FILE, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):  # Handle case where file exists but is not a list
                return {}
    except (json.JSONDecodeError, FileNotFoundError):
        return {}
    return Counter(data)



# === PASSWORD GATE ===
if not st.session_state.authenticated:
    # Create a container with top margin
    st.markdown("""
        <style>
            .login-container {
                margin-top: 100px;
                max-width: 600px;
                margin-left: auto;
                margin-right: auto;
                text-align: center;
                padding: 2rem;
            }
            .profile-pic {
                width: 100px;
                height: 100px;
                border-radius: 50%;
                margin: 0 auto 1rem;
                display: block;
                border: 3px solid #f0f2f6;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Main container with centered content
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # YouTube channel profile picture
    st.markdown(
        f'<img src="{PROFILE_PICTURE_URL}" ' 
        'class="profile-pic" alt="YouTube Channel Profile Picture">', 
        unsafe_allow_html=True
    )
    
    # Title and description
    st.markdown(f"<h2 style='margin-bottom: 1rem;'>🔐 {SITE_TITLE}</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='margin-bottom: 1.5rem; color: #666;'>"
        f"This tool is available <strong>free to registered users</strong>. "
        f"<a href='{REGISTRATION_URL}' target='_blank'>Register here</a>"
        f"</p>", 
        unsafe_allow_html=True
    )
    
    # Password input
    password = st.text_input("Enter your password:", type="password")
    
    if password == PASSWORD:
        st.session_state.authenticated = True
        log_access()
        st.success("Access granted! Welcome.")
        time.sleep(2)
        st.rerun()
    elif password:
        st.error("Incorrect password. Please try again.")
        st.stop()
    else:
        st.stop()
        
    # Close container
    st.markdown('</div>', unsafe_allow_html=True)

# === OPENAI SETUP ===
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# === DOWNLOAD AFTER AUTHENTICATION ===
if st.session_state.authenticated:
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

def timestamp_to_seconds(timestamp):
    """Convert timestamp format (e.g., "1:23:45" or "1:23") to seconds"""
    if not timestamp:
        return 0
        
    try:
        parts = timestamp.split(":")
        if len(parts) == 3:  # hours:minutes:seconds
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:  # minutes:seconds
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 1 and parts[0].isdigit():  # seconds
            return int(parts[0])
        return 0
    except Exception:
        return 0
        
def get_youtube_thumbnail_url(video_id):
    """Get the thumbnail URL for a YouTube video"""
    if not video_id:
        return ""
    # Use the maxresdefault image when available (highest quality)
    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    
def search_faiss(query_vector, top_k):
    # Request more results than needed to account for potential duplicates
    expanded_k = min(top_k * 5, len(metadata))  # Request 5x but don't exceed total data size
    scores, indices = index.search(query_vector, expanded_k)
    
    # Get actual number of results returned (may be less than expanded_k)
    actual_results = min(expanded_k, len(indices[0]))
    
    # First pass: collect all valid results
    all_results = []
    for i in range(actual_results):
        idx = indices[0][i]
        if idx >= 0 and idx < len(metadata):
            all_results.append((scores[0][i], metadata[idx]))
    
    # Sort all results by score (descending)
    all_results.sort(key=lambda x: x[0], reverse=True)
    
    # Second pass: filter by video proximity and handle shorts
    filtered_results = []
    video_times = {}  # Keep track of included times by video ID
    shorts_ids = set()  # Keep track of shorts videos we've already included
    
    for score, meta in all_results:
        video_url = meta.get("video_url", "")
        timestamp = meta.get("timestamp", "")
        video_id = extract_youtube_video_id(video_url)
        
        # Skip if no video ID (shouldn't happen with properly formatted data)
        if not video_id:
            continue
            
        # Check if this is a Short based on metadata
        is_short = meta.get("short", "") == "YES"
        
        # For Shorts videos, only include each unique video ID once
        if is_short:
            if video_id in shorts_ids:
                continue
            shorts_ids.add(video_id)
            # For Shorts, we'll store the video in metadata but not use timestamp for filtering
            filtered_results.append((score, meta))
            # If we have enough results, stop
            if len(filtered_results) >= top_k:
                break
            continue
            
        # For regular videos, proceed with normal timestamp-based filtering
        time_seconds = timestamp_to_seconds(timestamp)
        
        # Check if this video ID is already in our results
        if video_id in video_times:
            # Check all timestamps for this video to see if any are within 60 seconds
            too_close = False
            for existing_time in video_times[video_id]:
                if abs(existing_time - time_seconds) <= 60:
                    too_close = True
                    break
            
            # Skip this result if too close to an existing result
            if too_close:
                continue
        else:
            # Initialize list for this video ID
            video_times[video_id] = []
        
        # Include this result and record its timestamp
        filtered_results.append((score, meta))
        video_times[video_id].append(time_seconds)
        
        # If we have enough results, stop
        if len(filtered_results) >= top_k:
            break
    
    return filtered_results

# === MAIN UI ===
st.markdown(f"<div style='display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;'><img src='https://f000.backblazeb2.com/file/megatransfer/vaults/AC_Logo_Small.png' style='height: 24px; width: auto;'><h3 style='margin: 0;'>{SITE_TITLE}</h3></div>", unsafe_allow_html=True)
query = st.text_input("Type a question or comment to find helpful, accurate responses.")

if not query:
    col1, col2 = st.columns([2, 3])
    with col1:
        st.info("Pick the Number of Responses.")
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
            st.success(f"Found {len(top_results)} matches for your query:")
            for idx, (sim, qa) in enumerate(top_results):
                try:
                    question = qa.get("question", "[No question]")
                    answer = qa.get("answer", "[No answer]")
                    title = qa.get("video_title", "untitled")
                    timestamp = qa.get("timestamp", "0:00")
                    url = qa.get("video_url", "#")
                    segment = qa.get("segment_title", "")
                    
                    # Check if this is a Short based on metadata
                    is_short = qa.get("short", "") == "YES"
                    
                    # For Shorts videos, remove timestamp from URL if present
                    if is_short and "&t=" in url:
                        url = url.split("&t=")[0]
                    elif is_short and "?t=" in url:
                        url = url.split("?t=")[0]

                    st.markdown("----")
                    
                    # Extract video ID for embedding
                    video_id = extract_youtube_video_id(url)
                    
                    # Display embedded YouTube player if video ID is available
                    if video_id:
                        # Convert timestamp to seconds for the player
                        start_time = timestamp_to_seconds(timestamp) if timestamp and not is_short else 0
                        st_player.st_player(
                            f"https://youtu.be/{video_id}" + (f"?t={start_time}" if start_time > 0 and not is_short else ""),
                            height=0,  # Auto height for responsive layout
                            playing=False,
                            controls=True
                        )
                    
                    if title.lower().strip() not in ["untitled", "untitled video", ""]:
                        if is_short:
                            st.markdown(f"📲 **Shorts: {title}**") 
                        else:
                            st.markdown(f"🎬 **{title}**")
                    else:
                        if is_short:
                            st.markdown(f"📲 **YouTube Short**")
                        else:
                            st.markdown(f"🎬 **Video**")
                        
                    # Display segment title if available
                    if segment and segment.lower().strip() not in ["", "untitled"]:
                        st.markdown(f"🗂️ **Segment:** {segment}")
                    
                    # Only show timestamp for non-Short videos
                    if not is_short:    
                        st.markdown(f"⏰ **Timestamp:** {timestamp}")
                        
                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")

                    if is_short:
                        button_text = "📋 Copy Shorts link"
                    else:
                        button_text = "📋 Copy link"
                        
                    components.html(f"""
                    <div>
                        <button onclick="navigator.clipboard.writeText('{url}'); this.innerText='✅ Copied!'; setTimeout(() => this.innerText='{button_text}', 2000);" style="cursor:pointer; padding:4px 10px; font-size:0.85rem; border:1px solid #ccc; border-radius:5px; background:#f9f9f9;">{button_text}</button>
                    </div>
                    """, height=40)

                    st.markdown(
                        f" **<span style='color:green;'>Semantic similarity: {sim:.3f}</span>**",
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.warning(f"⚠️ Error displaying result: {e}")

# === LOGOUT BUTTON ===
st.markdown("---")
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button("🔓 Logout"):
    st.session_state.authenticated = False
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# === FOOTER ===
usage = get_monthly_usage()
current_month = datetime.now().strftime("%Y-%m")
st.markdown(f"📊 **Logins this month:** `{usage.get(current_month, 0)}`")
st.markdown(f"Last Updated: {LAST_UPDATED}")
st.markdown("---")
st.markdown(f"<span><img src='https://f000.backblazeb2.com/file/megatransfer/vaults/AC_Logo_Small.png' style='height: 1em; width: auto; vertical-align: middle; margin-right: 5px;'><a href='https://askclips.com' style='text-decoration: none; color: inherit;'>Powered by AskClips.com</a></span>", unsafe_allow_html=True)
