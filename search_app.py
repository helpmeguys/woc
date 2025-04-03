import streamlit as st
import json
import numpy as np
import pandas as pd
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
import streamlit.components.v1 as components

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
    </style>
""", unsafe_allow_html=True)

# === EVENT LOGGING ===
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

# === ADMIN PANEL ===
query_page = st.query_params.get("page", "")

if query_page == "admin":
    st.title("📊 Admin Dashboard")

    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        admin_pw = st.text_input("Enter admin password:", type="password")
        if admin_pw == ADMIN_PASSWORD:
            st.session_state.admin_authenticated = True
            st.success("Admin access granted.")
            time.sleep(1)
            st.rerun()
        else:
            if admin_pw:
                st.error("Incorrect admin password.")
            st.stop()

    try:
        with open(ACTIVITY_LOG_FILE, "r") as f:
            activity_data = json.load(f)
    except FileNotFoundError:
        activity_data = []
        st.warning("Activity log file not found. No data to display.")

    if activity_data:
        data = {
            "date": [log["timestamp"].split('T')[0] for log in activity_data],
            "event": [log["event"] for log in activity_data]
        }
        df = pd.DataFrame(data)
        df['count'] = 1  # Add a counter for aggregation

        # Group by date and event type, then unstack for bar chart compatibility
        daily_events = df.groupby(['date', 'event']).count().unstack(fill_value=0)
        daily_events.columns = daily_events.columns.droplevel()  # Remove multi-level index

        # Display bar chart
        st.bar_chart(daily_events)

        st.markdown("---")
        st.markdown("[🔙 Back to App](./)")
    st.stop()

# Remaining sections of the script remain unchanged...

