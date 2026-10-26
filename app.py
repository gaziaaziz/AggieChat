import json
import uuid
import requests
import streamlit as st
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────

API_BASE   = os.environ.get("API_BASE", "http://localhost:8000")
STREAM_URL = f"{API_BASE}/chat/stream"
TITLES_URL = f"{API_BASE}/titles"
HEALTH_URL = f"{API_BASE}/health"

MAROON = "#500000"
CREAM  = "#FDFAF4"
GOLD   = "#C5A95A"

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chat with the Collections",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── STYLES ───────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&display=swap');

/* Force light mode regardless of system setting */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background-color: {CREAM} !important;
    color: #1a1008 !important;
}}

/* Typography */
html, body, [class*="css"], p, div, span, label {{
    font-family: 'Source Serif 4', Georgia, serif !important;
}}

/* Hide Streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {MAROON} !important;
}}
[data-testid="stSidebar"] > div {{
    background-color: {MAROON} !important;
}}
[data-testid="stSidebar"] * {{
    color: {CREAM} !important;
}}
[data-testid="stSidebar"] button {{
    background-color: rgba(197,169,90,0.2) !important;
    border: 1px solid rgba(197,169,90,0.45) !important;
    color: {GOLD} !important;
    border-radius: 7px !important;
    font-family: 'Source Serif 4', serif !important;
}}
[data-testid="stSidebar"] button:hover {{
    background-color: rgba(197,169,90,0.35) !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.15) !important;
}}

/* Chat messages */
[data-testid="stChatMessage"] {{
    background-color: transparent !important;
    border: none !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {{
    color: #1a1008 !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {{
    font-size: 0.95rem !important;
    line-height: 1.75 !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a {{
    color: {MAROON} !important;
    text-decoration: underline;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {{
    color: #1a1008 !important;
    background-color: rgba(80,0,0,0.06) !important;
}}

/* User avatar */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
    flex-direction: row-reverse !important;
}}

/* Chat input container background */
[data-testid="stBottom"], [data-testid="stBottom"] > div {{
    background-color: {CREAM} !important;
}}

/* Chat input */
[data-testid="stChatInput"] textarea {{
    background-color: white !important;
    color: #1a1008 !important;
    font-family: 'Source Serif 4', serif !important;
    font-size: 0.93rem !important;
}}
[data-testid="stChatInput"]:focus-within textarea {{
    box-shadow: 0 0 0 3px rgba(80,0,0,0.08) !important;
}}

/* Sources styling */
.sources-block {{
    margin-top: 0.75rem;
    padding: 0.7rem 1rem;
    background: rgba(80,0,0,0.04);
    border-left: 3px solid {GOLD};
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    line-height: 1.6;
}}
.sources-block .sources-label {{
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {MAROON};
    font-weight: 600;
    display: block;
    margin-bottom: 0.35rem;
}}
.sources-block a {{
    color: {MAROON} !important;
    text-decoration: underline;
    text-underline-offset: 2px;
}}

/* Sidebar title pill */
.title-pill {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    margin: 0.2rem 0;
    font-size: 0.83rem;
    color: {CREAM};
}}
.sidebar-section-label {{
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(253,250,244,0.45);
    margin: 1rem 0 0.4rem;
    display: block;
}}
.health-indicator {{
    font-size: 0.72rem;
    color: rgba(253,250,244,0.4);
    margin-top: 0.5rem;
}}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────

if "session_id"    not in st.session_state:
    st.session_state.session_id   = str(uuid.uuid4())
if "messages"      not in st.session_state:
    st.session_state.messages     = []   # {"role", "content", "sources"}
if "recents"       not in st.session_state:
    st.session_state.recents      = []   # [{"session_id", "preview", "messages"}]
if "is_streaming"  not in st.session_state:
    st.session_state.is_streaming = False

# ─── HELPERS ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_titles():
    try:
        r = requests.get(TITLES_URL, timeout=5)
        return [t["name"] for t in r.json().get("titles", [])]
    except Exception:
        return ["The Battalion", "The Texas Aggie", "The Daily Bulletin/Reveille"]

@st.cache_data(ttl=60)
def fetch_health():
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        return r.json()
    except Exception:
        return {"status": "unreachable", "chunks_indexed": 0}

def start_new_chat():
    if st.session_state.messages:
        first_user = next(
            (m["content"] for m in st.session_state.messages if m["role"] == "user"),
            None
        )
        if first_user:
            preview    = first_user[:52] + "…" if len(first_user) > 52 else first_user
            exists     = any(r["session_id"] == st.session_state.session_id for r in st.session_state.recents)
            if not exists:
                st.session_state.recents.insert(0, {
                    "session_id": st.session_state.session_id,
                    "preview":    preview,
                    "messages":   list(st.session_state.messages),
                })
                st.session_state.recents = st.session_state.recents[:10]

    st.session_state.session_id   = str(uuid.uuid4())
    st.session_state.messages     = []
    st.session_state.is_streaming = False

def load_recent(recent: dict):
    st.session_state.session_id = recent["session_id"]
    st.session_state.messages   = list(recent["messages"])

def render_sources(sources_md: str):
    """Render the sources block beneath an assistant message."""
    if not sources_md:
        return
    import re
    # Strip the **Sources** header — we render our own styled label
    lines = sources_md.strip().split("\n")
    lines = [l for l in lines if not l.strip().startswith("**Sources**")]
    inner = "\n".join(lines).strip()
    if inner:
        st.markdown(
            f'<div class="sources-block">'
            f'<span class="sources-label">Sources</span>'
            f'{inner}'
            f'</div>',
            unsafe_allow_html=True
        )

def stream_response(question: str):
    """
    Generator: yields (partial_prose, sources) tuples as tokens arrive.
    """
    prose   = []
    sources = ""

    with requests.post(
        STREAM_URL,
        json={"question": question, "session_id": st.session_state.session_id},
        stream=True,
        timeout=120
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            if data["type"] == "token":
                prose.append(data["content"])
                yield "".join(prose), sources

            elif data["type"] == "sources":
                sources = data["content"]

            elif data["type"] == "warning":
                prose.append(f"\n\n> {data['content']}")
                yield "".join(prose), sources

            elif data["type"] == "done":
                break

            elif data["type"] == "error":
                prose.append(f"\n\n⚠️ {data['content']}")
                yield "".join(prose), sources
                break

    yield "".join(prose), sources

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<div style="font-family: Playfair Display, Georgia, serif; '
        f'font-size: 1.2rem; font-weight: 700; color: white; '
        f'padding: 0.5rem 0 1rem; border-bottom: 1px solid rgba(255,255,255,0.15);">'
        f'📰 Chat with the Collections</div>',
        unsafe_allow_html=True
    )

    st.write("")
    if st.button("✦  New chat", use_container_width=True, key="new_chat_btn"):
        start_new_chat()
        st.rerun()

    # Archives
    st.markdown('<span class="sidebar-section-label">Archives</span>', unsafe_allow_html=True)
    titles = fetch_titles()
    for title in titles:
        st.markdown(f'<div class="title-pill">📄 {title}</div>', unsafe_allow_html=True)

    # Recents
    if st.session_state.recents:
        st.markdown('<span class="sidebar-section-label">Recents</span>', unsafe_allow_html=True)
        for i, recent in enumerate(st.session_state.recents):
            label = f"{'▸ ' if recent['session_id'] == st.session_state.session_id else '💬 '}{recent['preview']}"
            if st.button(label, key=f"recent_{i}", use_container_width=True):
                load_recent(recent)
                st.rerun()

    # Health
    st.markdown("---")
    health = fetch_health()
    if health["status"] == "ok":
        st.markdown(
            f'<div class="health-indicator">● {health["chunks_indexed"]:,} chunks indexed</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="font-size:0.72rem; color:#ff9999;">⚠ API unreachable</div>',
            unsafe_allow_html=True
        )

# ─── HEADER ───────────────────────────────────────────────────────────────────

st.markdown(
    f'<h1 style="font-family: Playfair Display, Georgia, serif; '
    f'font-size: 1.4rem; font-weight: 600; color: {MAROON}; '
    f'margin: 1.2rem 0 0.1rem; padding-bottom: 0.8rem; '
    f'border-bottom: 1px solid rgba(80,0,0,0.1);">'
    f'Texas A&M Historical Newspaper Archives</h1>'
    f'<p style="font-style: italic; font-size: 0.8rem; '
    f'color: rgba(80,0,0,0.45); margin-top: 0; margin-bottom: 1rem;">'
    f'Searching The Battalion · The Texas Aggie · The Daily Bulletin/Reveille</p>',
    unsafe_allow_html=True
)

# ─── EMPTY STATE ──────────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown(
        f'<div style="text-align:center; padding: 4rem 2rem; color: rgba(80,0,0,0.3);">'
        f'<div style="font-size: 2.5rem; margin-bottom: 1rem;">📜</div>'
        f'<div style="font-family: Playfair Display, serif; font-size: 1.3rem; '
        f'font-weight: 600; color: rgba(80,0,0,0.4); margin-bottom: 0.5rem;">Ask the Archives</div>'
        f'<div style="font-size: 0.88rem; max-width: 380px; margin: 0 auto; line-height: 1.6;">'
        f'Search across digitized historical newspapers from Texas A&M University. '
        f'Ask about people, events, campus life, or any topic covered in the archives.</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# ─── RENDER EXISTING MESSAGES ─────────────────────────────────────────────────

for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "📰"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

# ─── CHAT INPUT & STREAMING ───────────────────────────────────────────────────

question = st.chat_input(
    "Ask about the archives…",
    disabled=st.session_state.is_streaming
)

if question:
    # Show user message immediately
    st.session_state.messages.append({
        "role": "user", "content": question, "sources": ""
    })
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    # Stream assistant response
    with st.chat_message("assistant", avatar="📰"):
        prose_placeholder = st.empty()
        final_prose       = ""
        final_sources     = ""

        try:
            for partial_prose, partial_sources in stream_response(question):
                final_prose   = partial_prose
                final_sources = partial_sources
                prose_placeholder.markdown(final_prose + "▌")
        except Exception as e:
            final_prose = f"Sorry, something went wrong connecting to the API: {str(e)}"

        # Replace cursor with final text
        prose_placeholder.markdown(final_prose)

        # Render sources beneath
        if final_sources:
            render_sources(final_sources)

    # Commit to session state
    st.session_state.messages.append({
        "role":    "assistant",
        "content": final_prose,
        "sources": final_sources,
    })