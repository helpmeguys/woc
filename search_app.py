# gospel_app_with_login_and_bookmarks.py
import streamlit as st
import sqlite3
import hashlib
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from gtts import gTTS
from io import BytesIO
import urllib.parse
import streamlit.components.v1 as components
import time
import requests
import pandas as pd
from collections import Counter, defaultdict
import os

# === CONFIGURATION ===
SITE_TITLE = st.secrets["SITE_TITLE"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
EMBEDDINGS_FILE = "qa_embeddings.json"
DOWNLOAD_URL = st.secrets["DOWNLOAD_URL"]
ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
REGISTRATION_URL = st.secrets["REGISTRATION_URL"]
ACCESS_LOG_FILE = "access_log.json"
ACTIVITY_LOG_FILE = "activity_log.json"

# === FILES AND PAGE CONFIG ===
st.set_page_config(page_title=f"📖 {SITE_TITLE}", layout="centered")

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

# === LOGGING FUNCTIONS ===
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

# === DB SETUP ===
DB_FILE = "users.db"

def get_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    return conn

def create_tables():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS bookmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        question TEXT,
        answer TEXT,
        url TEXT,
        timestamp TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    conn.commit()
    conn.close()

create_tables()

# === ADMIN DASHBOARD ===
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

        chart = pd.pivot_table(grouped, index="month", columns="event", values="count", fill_value=0)
        st.bar_chart(chart)

        st.download_button(
            label="📥 Download Activity Log as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='activity_log.csv',
            mime='text/csv'
        )

    st.subheader("📆 Monthly Breakdown")
    monthly_counts = defaultdict(lambda: Counter())
    for log in activity_data:
        month = log["timestamp"][:7]
        monthly_counts[month][log["event"]] += 1
    for month in sorted(monthly_counts.keys(), reverse=True):
        st.markdown(f"### {month}")
        for event, count in monthly_counts[month].items():
            st.markdown(f"- **{event.replace('_', ' ').title()}**: {count}")

    st.markdown("---")
    st.markdown("[🔙 Back to App](./)")
    st.stop()

# === NAVIGATION & USER AUTH ===
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.username = ""

tab_home, tab_bookmarks = st.tabs(["🏠 Home", "⭐ Bookmarks"])

st.title(f"📖 {SITE_TITLE}")

if st.session_state.user_id:
    st.success(f"Logged in as {st.session_state.username}")
    if st.button("🚪 Logout"):
        st.session_state.user_id = None
        st.session_state.username = ""
        st.experimental_rerun()
else:
    st.markdown("---")
    mode = st.radio("Choose an option", ["Login", "Register"], horizontal=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if mode == "Login":
        if st.button("🔓 Login"):
            user_id = authenticate_user(username, password)
            if user_id:
                st.session_state.user_id = user_id
                st.session_state.username = username
                log_access()
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")
    elif mode == "Register":
        if st.button("📝 Register"):
            if register_user(username, password):
                st.success("Registration successful! Please log in.")
            else:
                st.error("Username already taken. Try a different one.")
    st.stop()

# === DATABASE FUNCTIONS ===
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and result[1] == hash_password(password):
        return result[0]
    return None

def register_user(username, password):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def add_bookmark(user_id, question, answer, url):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO bookmarks (user_id, question, answer, url, timestamp) VALUES (?, ?, ?, ?, ?)",
              (user_id, question, answer, url, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    

def get_bookmarks(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, question, answer, url, timestamp FROM bookmarks WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
    results = c.fetchall()
    conn.close()
    return results
            

ensure_embeddings_file()

@st.cache_data
def load_qa_embeddings():
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            item["embedding"] = np.array(item["embedding"])
        return data

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def embed_query(text: str):
    client = OpenAI(api_key=OPENAI_API_KEY)
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

# === Q&A INTERFACE ===
with tab_home:
    qa_data = load_qa_embeddings()
    query = st.text_input("Type your gospel question or thought:")
    top_k = st.slider("🔧 Number of Results", 1, 20, 5)

    if query:
        log_event("search", {"query": query})
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
            st.markdown(f"📖 *{section}* — **{title.strip()}**")
            st.markdown(f"<a href='{url}' target='_blank'>▶️ Watch from {timestamp}</a>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("📋 Copy Link", key=f"copy_{idx}"):
                    log_event("copy_link", {"url": url})
                    st.toast("✅ Link copied!")
            with col2:
                if st.button("📧 Email", key=f"email_{idx}"):
                    log_event("email_sent", {"url": url})
                    body_text = f"Question: {question}\n\nAnswer: {answer}\n\nWatch here: {url}"
                    mailto = f"mailto:?subject={urllib.parse.quote('Your Gospel Q&A Answer')}&body={urllib.parse.quote(body_text)}"
                    components.html(f"<script>window.location.href='{mailto}'</script>", height=0)
            with col3:
                if st.button("⭐ Bookmark", key=f"bookmark_{idx}"):
                    add_bookmark(st.session_state.user_id, question, answer, url)
                    st.success("Bookmarked!")

            with st.expander("🎧 Listen to this answer"):
                audio = generate_tts_audio(f"Question: {question}. Answer: {answer}")
                if audio:
                    st.audio(audio, format="audio/mp3")

# === USER BOOKMARKS ===
with tab_bookmarks:
    st.markdown("---")
    st.subheader("⭐ My Bookmarks")
    bookmarks = get_bookmarks(st.session_state.user_id)
    if bookmarks:
        for b_id, q, a, url, ts in bookmarks:
            with st.container():
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.markdown(f"🔗 [View Video]({url})")
                st.caption(f"Saved on: {ts}")
                if st.button("❌ Delete", key=f"del_{b_id}"):
                    delete_bookmark(b_id)
                    st.experimental_rerun()
                st.markdown("---")
        df = pd.DataFrame(bookmarks, columns=["ID", "Question", "Answer", "URL", "Timestamp"])
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Bookmarks as CSV", data=csv, file_name="my_bookmarks.csv", mime="text/csv")
    else:
        st.info("No bookmarks yet. Start asking questions and bookmarking answers!")


# === FOOTER ===
usage = get_monthly_usage()
current_month = datetime.now().strftime("%Y-%m")
st.markdown(f"📈 **Logins this month:** `{usage.get(current_month, 0)}`")


