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

# === CONFIGURATION ===
SITE_TITLE = os.environ.get("SITE_TITLE")
PASSWORD = os.environ.get("ACCESS_PASSWORD")
REGISTRATION_URL = os.environ.get("REGISTRATION_URL")
INDEX_FILE = "embeddings.index"
METADATA_FILE = "metadata.json"
INDEX_URL = os.environ.get("INDEX_URL")
META_URL = os.environ.get("META_URL")
LAST_UPDATED = os.environ.get("LAST_UPDATED")
ACCESS_LOG_FILE = "access_log.json"

# === STYLING CUSTOMIZATION ===
# Get color customization from environment variables with defaults
BG_COLOR = os.environ.get("BG_COLOR", "#ffffff")
TEXT_COLOR = os.environ.get("TEXT_COLOR", "#333333")
PRIMARY_COLOR = os.environ.get("PRIMARY_COLOR", "#FF4B4B")  # Streamlit default red
SECONDARY_COLOR = os.environ.get("SECONDARY_COLOR", "#4B8BFF")  # Streamlit default blue
LINK_COLOR = os.environ.get("LINK_COLOR", "#0366d6")  # GitHub-like blue
HEADER_COLOR = os.environ.get("HEADER_COLOR", "#1E1E1E")  # For headers/titles
ACCENT_COLOR = os.environ.get("ACCENT_COLOR", "#FFD166")  # Vibrant accent color
CARD_BG_COLOR = os.environ.get("CARD_BG_COLOR", "#ffffff")  # Card background
BORDER_RADIUS = os.environ.get("BORDER_RADIUS", "12px")  # Modern rounded corners
BOX_SHADOW = os.environ.get("BOX_SHADOW", "0 4px 20px rgba(0,0,0,0.08)")  # Subtle shadow

# Helper function to convert hex to RGB for CSS
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# === PAGE CONFIG ===
st.set_page_config(page_title=f"📖 {SITE_TITLE}", layout="centered", initial_sidebar_state="collapsed")

# === STYLING ===
st.markdown(f"""
    <style>
        /* Import modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Custom colors from environment variables */
        :root {{
            --background-color: {BG_COLOR};
            --text-color: {TEXT_COLOR};
            --primary-color: {PRIMARY_COLOR};
            --secondary-color: {SECONDARY_COLOR};
            --link-color: {LINK_COLOR};
            --header-color: {HEADER_COLOR};
            --accent-color: {ACCENT_COLOR};
            --card-bg-color: {CARD_BG_COLOR};
            --border-radius: {BORDER_RADIUS};
            --box-shadow: {BOX_SHADOW};
            --animation-speed: 0.2s;
            --spacing-xs: 0.25rem;
            --spacing-sm: 0.5rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;
            --spacing-xl: 2rem;
            --primary-rgb: {hex_to_rgb(PRIMARY_COLOR)[0]}, {hex_to_rgb(PRIMARY_COLOR)[1]}, {hex_to_rgb(PRIMARY_COLOR)[2]};
        }}
        
        /* Modern App Header */
        .app-header {{
            background-color: var(--card-bg-color);
            border-radius: var(--border-radius);
            padding: var(--spacing-md);
            margin-bottom: var(--spacing-lg);
            box-shadow: var(--box-shadow);
            border-top: 4px solid var(--primary-color);
            position: relative;
            overflow: hidden;
        }}
        
        .app-header::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(var(--primary-rgb), 0.05) 0%, transparent 50%);
            pointer-events: none;
        }}
        
        .app-title {{
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
            padding: 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        /* Loading Animation */
        @keyframes pulse {{
            0% {{ opacity: 0.6; transform: scale(0.98); }}
            50% {{ opacity: 1; transform: scale(1); }}
            100% {{ opacity: 0.6; transform: scale(0.98); }}
        }}
        
        .loading-animation {{
            animation: pulse 1.5s infinite ease-in-out;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin: var(--spacing-md) 0;
            padding: var(--spacing-lg);
            background-color: rgba(var(--primary-rgb), 0.05);
            border-radius: var(--border-radius);
        }}
        
        /* 2025 Modern Typography System */
        body, .stMarkdown, p, span, li, div, button, input, textarea, select {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            letter-spacing: -0.011em;
        }}
        
        /* Apply custom colors with enhanced styling */
        .stApp {{
            background-color: var(--background-color) !important;
        }}
        
        /* Modern Typography Styling */
        .stMarkdown, p, span, li, div, .stTextInput > label {{
            color: var(--text-color) !important;
            font-weight: 400;
            line-height: 1.6;
            font-size: 1rem;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: var(--header-color) !important;
            font-weight: 600;
            letter-spacing: -0.022em;
            line-height: 1.2;
        }}
        
        h1 {{font-size: 2.5rem;}}
        h2 {{font-size: 2rem;}}
        h3 {{font-size: 1.75rem;}}
        h4 {{font-size: 1.5rem;}}
        h5 {{font-size: 1.25rem;}}
        h6 {{font-size: 1rem;}}
        
        /* Enhanced Links with Hover Effects */
        a, a:visited {{
            color: var(--link-color) !important;
            text-decoration: none;
            font-weight: 500;
            transition: all var(--animation-speed) ease;
            border-bottom: 1px solid transparent;
        }}
        
        a:hover {{
            border-bottom: 1px solid var(--link-color);
        }}
        
        /* Modern Card Design System */
        .search-result {{
            background-color: var(--card-bg-color);
            border-radius: var(--border-radius);
            padding: var(--spacing-md);
            margin-bottom: var(--spacing-lg);
            box-shadow: var(--box-shadow);
            transition: transform var(--animation-speed) ease, box-shadow var(--animation-speed) ease;
            border: 1px solid rgba(0,0,0,0.05);
            overflow: hidden;
        }}
        
        .search-result:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(0,0,0,0.1);
        }}
        
        /* Modern UI Components */
        .stButton>button {{
            background-color: var(--primary-color) !important;
            color: white !important;
            border-radius: var(--border-radius) !important;
            font-weight: 500;
            border: none;
            padding: 0.6em 1.2em;
            transition: all var(--animation-speed) ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        
        .stButton>button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        /* Modern Sliders */
        .stSlider>div>div>div>div {{
            background-color: var(--primary-color) !important;
        }}
        
        /* Search Input Enhancement */
        .stTextInput input {{
            border-radius: var(--border-radius) !important;
            border: 1px solid rgba(0,0,0,0.1);
            padding: 0.75rem 1rem !important;
            font-size: 1rem !important;
            box-shadow: var(--box-shadow) !important;
            transition: all var(--animation-speed) ease;
        }}
        
        .stTextInput input:focus {{
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 2px rgba(var(--primary-color-rgb), 0.2) !important;
        }}
        
        /* Content Containers */
        .metadata-container {{
            display: flex;
            flex-wrap: wrap;
            gap: var(--spacing-sm);
            margin-top: var(--spacing-sm);
            margin-bottom: var(--spacing-sm);
        }}
        
        .metadata-container > * {{
            margin-right: var(--spacing-md);
        }}
        
        .qa-content {{
            margin-top: var(--spacing-md);
            margin-bottom: var(--spacing-md);
            border-left: 4px solid var(--primary-color);
            padding: var(--spacing-sm) var(--spacing-md);
            background-color: rgba(0,0,0,0.02);
            border-radius: 0 var(--border-radius) var(--border-radius) 0;
        }}
        
        /* Video Container Enhancement */
        .video-container {{
            border-radius: var(--border-radius);
            overflow: hidden;
            box-shadow: var(--box-shadow);
            margin-bottom: var(--spacing-md);
        }}
        
        /* Horizontal Dividers */
        hr {{
            border: none;
            height: 1px;
            background: linear-gradient(to right, transparent, rgba(0,0,0,0.1), transparent);
            margin: var(--spacing-md) 0;
        }}
        
        /* Hide deployment elements */
        .stDeployButton, [data-testid="stStatusWidget"], .viewerBadge_container__1QSob,
        .stActionButtonIcon, div[class*="floating"], [data-testid="collapsedControl"] {{
            display: none !important;
        }}
        
        /* Enhanced Buttons */
        .copy-button, .youtube-button {{
            border-radius: var(--border-radius) !important;
            transition: all var(--animation-speed) ease !important;
            font-weight: 500 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.5rem !important;
        }}
        
        .copy-button:hover, .youtube-button:hover {{
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        }}
        
        /* Responsive design for mobile devices */
        @media screen and (max-width: 640px) {{
            .block-container {{
                padding-top: var(--spacing-sm) !important;
                padding-bottom: var(--spacing-md) !important;
                padding-left: var(--spacing-xs) !important;
                padding-right: var(--spacing-xs) !important;
            }}
            
            /* Adjust card padding on mobile */
            .search-result {{
                padding: var(--spacing-sm);
                margin-bottom: var(--spacing-md);
            }}
            
            /* Reduce spacing between elements */
            p, h1, h2, h3, h4, h5, h6 {{
                margin-top: var(--spacing-xs) !important;
                margin-bottom: var(--spacing-xs) !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
            }}
            
            /* Reduce spacing for horizontal lines */
            hr {{
                margin: var(--spacing-sm) 0 !important;
            }}
            
            /* Compact results container */
            .stAlert, .stMarkdown, .element-container {{
                margin-top: var(--spacing-xs) !important;
                margin-bottom: var(--spacing-xs) !important;
                padding-top: var(--spacing-xs) !important;
                padding-bottom: var(--spacing-xs) !important;
            }}
            
            /* Reduce font sizes on mobile */
            h1 {{font-size: 1.8rem !important;}}
            h2 {{font-size: 1.5rem !important;}}
            h3 {{font-size: 1.3rem !important;}}
            h4 {{font-size: 1.2rem !important;}}
            h5 {{font-size: 1.1rem !important;}}
            h6 {{font-size: 1rem !important;}}
            
            /* Stack buttons on mobile */
            .button-container {{
                display: flex;
                flex-direction: column;
                gap: var(--spacing-xs);
            }}
        }}
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
    st.markdown(f"""<div class='app-header'>
        <h3 class='app-title'>🔐 {SITE_TITLE}</h3>
        <p>Welcome to our semantic search engine</p>
    </div>""", unsafe_allow_html=True)
    
    st.markdown(f"""<div style='background-color:var(--card-bg-color); padding:var(--spacing-md); 
        border-radius:var(--border-radius); box-shadow:var(--box-shadow);'>
        This tool is available <strong>free to registered users</strong>.<br>
        <a href='{REGISTRATION_URL}' target='_blank' style='display:inline-flex; align-items:center; margin-top:10px;
        gap:5px; color:var(--link-color); text-decoration:none;'>📝 Click here to register
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" 
        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
        <polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
        </a>
    </div>""", unsafe_allow_html=True)
    
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
    
def get_youtube_embed_html(video_id, timestamp=None, is_short=False):
    """Generate HTML to embed a YouTube video with optional timestamp
    
    For YouTube Shorts, we ignore the timestamp to ensure the entire short plays.
    """
    if not video_id:
        return ""
    
    # Convert timestamp to seconds if provided and not a short
    start_seconds = 0
    if timestamp and not is_short:
        start_seconds = timestamp_to_seconds(timestamp)
            
    # Create appropriate embed URL based on video type and timestamp
    if is_short:
        # For shorts videos
        embed_url = f"https://www.youtube.com/embed/{video_id}?rel=0&modestbranding=1"
    else:
        # For regular videos with optional timestamp
        embed_url = f"https://www.youtube.com/embed/{video_id}?start={start_seconds}&rel=0&modestbranding=1"
    
    # Return complete HTML with container and iframe
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
st.markdown(f"""<div class='app-header'>
    <h3 class='app-title'>🔎 {SITE_TITLE}</h3>
    <p>Search through video content using AI semantic matching</p>
</div>""", unsafe_allow_html=True)

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

    with st.spinner("""<div class='loading-animation'>
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" 
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg> Searching by meaning...
        </div>"""):
        query_vec = embed_query(query)
        top_results = search_faiss(query_vec, top_k)

def timestamp_to_seconds(timestamp):
    """Convert a timestamp string (HH:MM:SS, MM:SS, or SS) to seconds."""
    if not timestamp:
        return 0
        
    parts = timestamp.split(":")
    if len(parts) == 3:  # hours:minutes:seconds
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:  # minutes:seconds
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 1 and parts[0].isdigit():  # seconds
        return int(parts[0])
    return 0

# Main search function
def search_and_display_results(query, top_k=5):
    """Process search query and display results"""
    if not query.strip():
        return
        
    with st.spinner("Searching by meaning..."):
        try:
            # Start timing the search process
            start_time = time.time()
            
            # Get query vector
            query_vector = embed_query(query)
            
            # Search in FAISS index
            results = search_faiss(query_vector, top_k)
            
            # Calculate elapsed time and show in sidebar
            elapsed = time.time() - start_time
            st.sidebar.info(f"⏱️ Query processed in {elapsed:.2f} seconds")
            
            # Display results
            if not results:
                st.warning("⚠️ No results found for your query. Please try a different question.")
                return
                
            # Show success message with count
            st.success(f"Found {len(results)} matches for your query:")
            
            # Process and display each result
            for idx, (score, item) in enumerate(results):
                try:
                    # Extract information from the result
                    question = item.get("question", "[No question]")
                    answer = item.get("answer", "[No answer]")
                    url = item.get("url", "")
                    timestamp = item.get("timestamp", "0:00")
                    title = item.get("video_title", "untitled")
                    segment = item.get("segment", "")
                    
                    # Check if this is a Short based on metadata
                    is_short = item.get("short", "") == "YES"
                    
                    # For Shorts videos, remove timestamp from URL if present
                    if is_short and "&t=" in url:
                        url = url.split("&t=")[0]
                    elif is_short and "?t=" in url:
                        url = url.split("?t=")[0]
                    
                    # Extract video ID for embedding
                    video_id = extract_youtube_video_id(url)
                    
                    # Prepare title display
                    if title.lower().strip() not in ["untitled", "untitled video", ""]:
                        if is_short:
                            title_display = f"Shorts: {title}"
                        else:
                            title_display = title
                    else:
                        if is_short:
                            title_display = "YouTube Short"
                        else:
                            title_display = "Video"
                    
                    # Convert timestamp to seconds for embed URL
                    start_seconds = 0
                    if timestamp and not is_short:
                        start_seconds = timestamp_to_seconds(timestamp)
                            
                    # Create embed URL with appropriate parameters
                    if is_short:
                        embed_url = f"https://www.youtube.com/embed/{video_id}?rel=0&modestbranding=1"
                    else:
                        embed_url = f"https://www.youtube.com/embed/{video_id}?start={start_seconds}&rel=0&modestbranding=1"
                    
                    # Generate separate button HTML to avoid f-string nesting issues
                    copy_button_html = f"""
                    <button onclick="
                        navigator.clipboard.writeText('{url}');
                        this.innerHTML = '<svg xmlns=\'http://www.w3.org/2000/svg\' width=\'14\' height=\'14\' viewBox=\'0 0 24 24\' fill=\'none\' stroke=\'currentColor\' stroke-width=\'2\' stroke-linecap=\'round\' stroke-linejoin=\'round\'><path d=\'M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2\'></path><rect x=\'8\' y=\'2\' width=\'8\' height=\'4\' rx=\'1\' ry=\'1\'></rect></svg> <span style=\'margin-left:4px;\'>Copied!</span>';
                        setTimeout(() => {{
                            this.innerHTML = '<svg xmlns=\'http://www.w3.org/2000/svg\' width=\'14\' height=\'14\' viewBox=\'0 0 24 24\' fill=\'none\' stroke=\'currentColor\' stroke-width=\'2\' stroke-linecap=\'round\' stroke-linejoin=\'round\'><path d=\'M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2\'></path><rect x=\'8\' y=\'2\' width=\'8\' height=\'4\' rx=\'1\' ry=\'1\'></rect></svg> <span style=\'margin-left:4px;\'>{"Copy Shorts link" if is_short else "Copy link"}</span>';
                        }}, 2000);
                        " 
                        style="cursor: pointer; padding: 8px 16px; font-size: 0.9rem; border: 1px solid rgba(0,0,0,0.15); 
                        border-radius: {BORDER_RADIUS}; background: #ffffff; box-shadow: {BOX_SHADOW};">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
                            <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
                        </svg>
                        <span style="margin-left: 4px;">{"Copy Shorts link" if is_short else "Copy link"}</span>
                    </button>
                    """
                    
                    # Generate the HTML for the entire result card with timestamp on separate line
                    result_html = f"""
                    <div style="background-color: {CARD_BG_COLOR}; border-radius: {BORDER_RADIUS}; padding: 16px; margin-bottom: 24px; box-shadow: {BOX_SHADOW};">
                        <!-- Video Title -->
                        <div style="margin-bottom: 12px;">
                            <h3 style="margin-bottom: 8px; font-weight: 600; color: {HEADER_COLOR};">
                                {"📱" if is_short else "📖"} {title_display}
                            </h3>
                        </div>
                        
                        <!-- Segment info (on its own line) -->
                        {('<p style="margin: 8px 0;"><strong>Segment:</strong> ' + segment + '</p>') if segment and segment.lower().strip() not in ["", "untitled"] else ''}
                        
                        <!-- Timestamp (on its own line) -->
                        {('<p style="margin: 8px 0; color: ' + PRIMARY_COLOR + ';"><strong>⏰ Timestamp:</strong> ' + timestamp + '</p>') if timestamp and not is_short else ''}
                        
                        <!-- YouTube video embed with proper aspect ratio -->
                        <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; border-radius: {BORDER_RADIUS}; margin: 16px 0;">
                            <iframe 
                                src="{embed_url}" 
                                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0;" 
                                allowfullscreen 
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture">
                            </iframe>
                        </div>
                        
                        <!-- Q&A content -->
                        <div style="margin: 16px 0; background-color: rgba(0,0,0,0.02); padding: 12px; border-radius: {BORDER_RADIUS};">
                            <p style="margin-bottom: 8px;"><strong>Q:</strong> {question}</p>
                            <p><strong>A:</strong> {answer}</p>
                        </div>
                        
                        <!-- Button container -->
                        <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; align-items: center;">
                            <!-- Copy link button (generated separately to avoid f-string nesting) -->
                            {copy_button_html}
                            
                            <!-- Open in YouTube button -->
                            <a href="{url}" target="_blank" 
                                style="display: inline-block; padding: 8px 16px; font-size: 0.9rem; border: 1px solid rgba(255,0,0,0.7); 
                                border-radius: {BORDER_RADIUS}; background: #ffffff; color: #FF0000; text-decoration: none; box-shadow: {BOX_SHADOW};">
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="#FF0000">
                                    <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/>
                                </svg>
                                <span style="margin-left: 4px;">Open in YouTube</span>
                            </a>
                            
                            <!-- Similarity score -->
                            <div style="margin-left: auto;">
                                <span style="background-color: rgba({hex_to_rgb(PRIMARY_COLOR)[0]}, {hex_to_rgb(PRIMARY_COLOR)[1]}, {hex_to_rgb(PRIMARY_COLOR)[2]}, 0.1); padding: 4px 8px; border-radius: 20px; font-size: 0.8rem;">
                                    <span style="color: {PRIMARY_COLOR}; font-weight: 500;">Similarity: {score:.3f}</span>
                                </span>
                            </div>
                        </div>
                    </div>
                    """
                    
                    # Render the result using components.html
                    # Use a taller height for proper iframe display
                    components.html(result_html, height=650, scrolling=True)
                    
                except Exception as e:
                    st.warning(f"⚠️ Error displaying result {idx}: {str(e)}")
                    st.exception(e)
        
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            st.exception(e)

# Call the search and display function with the query
if query:
    search_and_display_results(query, top_k)

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

st.markdown(f"""<div style='background-color:var(--card-bg-color); padding:var(--spacing-md); border-radius:var(--border-radius); box-shadow:var(--box-shadow); margin-top:var(--spacing-lg);'>
    <p style='display:flex; justify-content:space-between; margin-bottom:var(--spacing-xs);'>
        <span>📊 <strong>Logins this month:</strong> {usage.get(current_month, 0)}</span>
        <span>🔄 {LAST_UPDATED}</span>
    </p>
    <hr style='opacity:0.2; margin:var(--spacing-xs) 0;'>
    <div style='display:flex; justify-content:center; align-items:center;'>
        <a href='https://askclips.com' target='_blank' style='display:flex; align-items:center; gap:var(--spacing-xs);'>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" 
            stroke="var(--primary-color)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="16" x2="12" y2="12"></line>
            <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            <span><strong>Powered by AskClips.com</strong></span>
        </a>
    </div>
</div>""", unsafe_allow_html=True)
