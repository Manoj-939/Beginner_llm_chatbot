
import os
 
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
 
# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
 
st.set_page_config(page_title="Gemini Chat", page_icon="✨", layout="centered")
 
PERSONAS = {
    "Friendly helper": "You are a warm, concise assistant. Keep answers clear and easy to read.",
    "Coding buddy": "You are an expert programming mentor. Give working code with short explanations.",
    "Explain it simply": "Explain everything in simple words with everyday analogies.",
}
 
SUGGESTIONS = [
    "Explain REST vs GraphQL in simple words",
    "Quiz me on JavaScript closures",
    "Write a Python function to check palindromes",
]
 
# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    persona = st.selectbox("Style", list(PERSONAS.keys()))
    temperature = st.slider("Creativity", 0.0, 1.5, 0.7, 0.1)
    effects = st.toggle("Visual effects", value=True)
    MODEL = st.selectbox("Model", ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"])
    st.caption(f"Model: `{MODEL}`")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.toast("Chat cleared", icon="🧹")
        st.rerun()
 
# ---------------------------------------------------------------------------
# Styling and effects
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Sora:wght@600;700&display=swap');
 
html, body, .stApp { font-family: 'DM Sans', sans-serif; }
 
/* Slowly drifting background */
.stApp {
    background: linear-gradient(-45deg, #0f1226, #1b1f4b, #0e3b4a, #2a1647);
    background-size: 400% 400%;
    animation: drift 28s ease infinite;
}
@keyframes drift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
header[data-testid="stHeader"] { background: transparent; }
 
/* Shimmering title */
.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(90deg, #7dd3fc, #f0abfc, #7dd3fc);
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: shine 6s linear infinite;
}
@keyframes shine { to { background-position: 200% center; } }
.hero-sub { color: #b8c0e8; margin: 0.2rem 0 1.4rem 0; }
 
/* Glass chat bubbles that rise into place */
div[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 18px;
    padding: 1rem 1.2rem;
    animation: rise 0.45s ease both;
}
@keyframes rise {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
 
/* Glowing input */
div[data-testid="stChatInput"] { border-radius: 16px; transition: box-shadow 0.25s ease; }
div[data-testid="stChatInput"]:focus-within {
    box-shadow: 0 0 0 2px rgba(125, 211, 252, 0.55), 0 0 28px rgba(240, 171, 252, 0.25);
}
 
/* Typing indicator */
.typing span {
    display: inline-block;
    width: 8px; height: 8px;
    margin-right: 6px;
    border-radius: 50%;
    background: #7dd3fc;
    animation: bounce 1.2s infinite ease-in-out;
}
.typing span:nth-child(2) { animation-delay: 0.15s; background: #c4b5fd; }
.typing span:nth-child(3) { animation-delay: 0.30s; background: #f0abfc; }
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
    30%           { transform: translateY(-7px); opacity: 1; }
}
 
@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
if not effects:
    st.markdown("<style>*{animation:none !important;}</style>", unsafe_allow_html=True)
 
TYPING_HTML = '<div class="typing"><span></span><span></span><span></span></div>'
 
st.markdown('<h1 class="hero-title">Gemini Chat</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Ask anything. Powered by the Gemini free tier.</p>', unsafe_allow_html=True)
 
# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
if not API_KEY:
    st.error(
        "No API key found. Add `GEMINI_API_KEY=your_key` to the `.env` file "
        "(get a free key at https://aistudio.google.com/apikey) and restart the app."
    )
    st.stop()
 
 
@st.cache_resource
def get_client(key: str) -> genai.Client:
    return genai.Client(api_key=key)
 
 
client = get_client(API_KEY)
 
 
def to_contents(messages):
    """Convert stored messages into Gemini's content format."""
    return [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
    ]
 
 
def stream_reply(messages):
    config = types.GenerateContentConfig(
        system_instruction=PERSONAS[persona],
        temperature=temperature,
    )
    for chunk in client.models.generate_content_stream(
        model=MODEL, contents=to_contents(messages), config=config
    ):
        if chunk.text:
            yield chunk.text
 
 
# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🙂" if m["role"] == "user" else "✨"):
        st.markdown(m["content"])
 
# Suggestion buttons on an empty chat
if not st.session_state.messages:
    cols = st.columns(len(SUGGESTIONS))
    for col, text in zip(cols, SUGGESTIONS):
        if col.button(text, use_container_width=True):
            st.session_state.queued = text
            st.rerun()
 
prompt = st.chat_input("Type your message...") or st.session_state.pop("queued", None)
 
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(prompt)
 
    with st.chat_message("assistant", avatar="✨"):
        box = st.empty()
        box.markdown(TYPING_HTML, unsafe_allow_html=True)
        reply = ""
        try:
            for piece in stream_reply(st.session_state.messages):
                reply += piece
                box.markdown(reply + "▌")  # blinking-cursor feel while streaming
            box.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.session_state.messages.pop()  # drop the unanswered prompt
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                box.warning("Free-tier rate limit reached. Wait a minute and try again.")
            elif "API key" in msg or "403" in msg or "401" in msg:
                box.error("The API key was rejected. Check GEMINI_API_KEY in your .env file.")
            else:
                box.error(f"Something went wrong: {msg}")
 
    if effects and "thank" in prompt.lower():
        st.balloons()