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
from io import BytesIO
from PIL import Image
import re

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
@st.cache_data(ttl=3600)  # Cache for 1 hour
def is_youtube_short(video_id):
    """Determine if a video is a YouTube Short based on its thumbnail dimensions and duration
    
    YouTube Shorts typically have:
    1. Portrait orientation (height > width)
    2. Duration of less than 180 seconds (3 minutes)
    """
    if not video_id:
        return False
        
    # First check if the URL pattern contains "shorts" (fast path)
    # This is handled elsewhere but mentioned for clarity
    
    # Check thumbnail dimensions
    width, height = get_thumbnail_dimensions(video_id)
    if width is None or height is None:
        # If we can't get dimensions, fall back to duration only
        is_portrait = False
    else:
        is_portrait = height > width
        
    # Check duration
    duration = estimate_video_duration(video_id)
    is_short_duration = duration is not None and duration <= 180
    
    # A video is considered a Short if it has portrait orientation AND short duration
    return is_portrait and is_short_duration

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
        elif "youtube.com/shorts/" in url:
            # Handle youtube.com/shorts/VIDEO_ID format
            return url.split("youtube.com/shorts/")[1].split("?")[0]
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

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_thumbnail_dimensions(video_id):
    """Get the dimensions of a YouTube video thumbnail"""
    if not video_id:
        return None, None
        
    # Try to get the highest quality thumbnail first
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    try:
        response = requests.get(thumbnail_url, stream=True, timeout=5)
        if response.status_code != 200:
            # Fall back to standard quality if max is not available
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
            response = requests.get(thumbnail_url, stream=True, timeout=5)
            if response.status_code != 200:
                return None, None
                
        img = Image.open(BytesIO(response.content))
        return img.width, img.height
    except Exception as e:
        print(f"Error getting thumbnail dimensions: {e}")
        return None, None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def estimate_video_duration(video_id):
    """Estimate the duration of a YouTube video
    
    This is a basic implementation that tries to fetch video metadata from YouTube.
    Note: For a production app, you might want to use the YouTube API instead.
    """
    if not video_id:
        return None
        
    try:
        # Try to get the video page and extract duration using regex
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(video_url, timeout=5)
        if response.status_code == 200:
            # Look for the lengthSeconds pattern in the page
            match = re.search(r'"lengthSeconds":"(\d+)"', response.text)
            if match:
                return int(match.group(1))
                
            # Alternative pattern
            match = re.search(r'"approxDurationMs":"(\d+)"', response.text)
            if match:
                return int(match.group(1)) // 1000
    except Exception as e:
        print(f"Error estimating video duration: {e}")
    
    return None
    
def get_youtube_embed_html(video_id, timestamp=None, is_shorts=False):
    """Generate HTML to embed a YouTube video with optional timestamp"""
    if not video_id:
        return ""
    
    # For YouTube Shorts, we skip adding the timestamp to ensure the entire short plays
    start_seconds = 0
    if timestamp and not is_shorts:
        parts = timestamp.split(":")
        if len(parts) == 3:  # hours:minutes:seconds
            start_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:  # minutes:seconds
            start_seconds = int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 1 and parts[0].isdigit():  # seconds
            start_seconds = int(parts[0])
            
    # Create an embedded YouTube player with autoplay disabled and modest branding
    # The ?start parameter specifies where to start the video in seconds (skipped for Shorts)
    embed_url = f"https://www.youtube.com/embed/{video_id}?rel=0&modestbranding=1"
    if not is_shorts and start_seconds > 0:
        embed_url += f"&start={start_seconds}"
    
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
    
    # Second pass: filter by video proximity (60 second window)
    filtered_results = []
    video_times = {}  # Keep track of included times by video ID
    shorts_ids = set()  # Keep track of YouTube Shorts videos we've already included
    
    for score, meta in all_results:
        video_url = meta.get("video_url", "")
        timestamp = meta.get("timestamp", "")
        video_id = extract_youtube_video_id(video_url)
        
        # Skip if no video ID (shouldn't happen with properly formatted data)
        if not video_id:
            continue
        
        # Check if this is a YouTube Shorts video - first check URL pattern (fast path)
        is_shorts = "youtube.com/shorts/" in video_url
        
        # If not detected by URL pattern, check thumbnail dimensions and duration
        if not is_shorts and video_id:
            is_shorts = is_youtube_short(video_id)
        
        # For Shorts videos, only include each unique video ID once
        if is_shorts:
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
                    
                    # Check if this is a YouTube Shorts video - first check URL pattern (fast path)
                    is_shorts = "youtube.com/shorts/" in url
                    
                    # If not detected by URL pattern, check thumbnail dimensions and duration
                    if not is_shorts and video_id:
                        is_shorts = is_youtube_short(video_id)
                    
                    # Display embedded YouTube player if video ID is available
                    if video_id:
                        embed_html = get_youtube_embed_html(video_id, timestamp, is_shorts)
                        components.html(embed_html, height=400)
                    
                    if title.lower().strip() not in ["untitled", "untitled video", ""]:
                        if is_shorts:
                            st.markdown(f"📲 **Shorts: {title}**")
                        else:
                            st.markdown(f"📖 **{title}**")
                    else:
                        if is_shorts:
                            st.markdown(f"📲 **YouTube Short**")
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
