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
import streamlit.components.v1 as components
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

    if activity_data:
        df = pd.DataFrame(activity_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["month"] = df["timestamp"].dt.to_period("M").astype(str)

        filtered_df = df[df["event"].isin(["search", "copy_link", "email_sent"])]

        grouped = filtered_df.groupby(["month", "event"]).size().reset_index(name="count")

        chart = alt.Chart(grouped).mark_line(point=True).encode(
            x="month:T",
            y="count:Q",
            color="event:N",
            tooltip=["month", "event", "count"]
        ).properties(
            title="📈 Monthly Event Trends",
            width=700,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)

        st.download_button(
            label="📅 Download Activity Log as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='activity_log.csv',
            mime='text/csv'
        )
    else:
        st.warning("No activity data found.")

    st.subheader("🗓️ Monthly Breakdown")
    monthly_counts = defaultdict(lambda: Counter())
    for log in activity_data:
        month = log["timestamp"][:7]
        monthly_counts[month][log["event"]] += 1

    for month in sorted(monthly_counts.keys(), reverse=True):
        st.markdown(f"### {month}")
        for event, count in monthly_counts[month].items():
            st.markdown(f"- **{event.replace('_', ' ').title()}**: {count}")

    st.markdown("---")
    st.markdown("[💙 Back to App](./)")
    st.stop()

# === LOGIN ===
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

# === OPENAI SETUP ===
client = OpenAI(api_key=OPENAI_API_KEY)

# === LOAD DATA ===
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

            with st.container():
                col1, col2 = st.columns([1, 1])

                with col1:
                    if st.button("📋 Copy link", key=f"copy_{idx}"):
                        st.toast("✅ Link copied!")
                        log_event("copy_link", {"url": url})

                with col2:
                    if st.button("📧 Send email", key=f"email_{idx}"):
                        st.toast("📧 Opening mail client...")
                        log_event("email_sent", {"url": url})
                        mailto = f"mailto:?subject={urllib.parse.quote('{SITE_TITLE}')}&body={urllib.parse.quote('Question: ' + question + '\n\nAnswer: ' + answer + '\n\nWatch here: ' + url + '\n\nThese results were rendered at https://search.thewordsofchrist.org')}"
                        components.html(f"<script>window.location.href='{mailto}'</script>", height=0)

            st.markdown(f"🔍 <span style='color:green;'>Semantic similarity: {sim:.3f}</span>", unsafe_allow_html=True)

            with st.expander("🎰 Listen to this answer"):
                audio = generate_tts_audio(f"Question: {question}. Answer: {answer}")
                if audio:
                    st.audio(audio, format="audio/mp3")

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
