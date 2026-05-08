import streamlit as st
from rag import initialize_rag_system, search_bank_answer
from deep_translator import GoogleTranslator
# ---------------- CONFIG ----------------
st.set_page_config(page_title="Banking AI Assistant", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>

/* REMOVE ALL EXTRA SPACE */
.block-container {
    padding: 0rem !important;
}

header, footer {
    display: none !important;
}

section.main > div {
    padding-top: 0rem !important;
}

/* REMOVE WHITE GAP */
html, body {
    margin: 0;
    padding: 0;
}

/* BACKGROUND */
[data-testid="stAppViewContainer"]{
    background: linear-gradient(to right,#0f2027,#203a43,#2c5364);
}

/* SIDEBAR */
section[data-testid="stSidebar"]{
    background: linear-gradient(to bottom,#08131f,#10263d);
}

/* TITLE */
.title{
    text-align:center;
    font-size:45px;
    color:#00ffd5;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:white;
    font-size:18px;
}

/* CARDS */
.card{
    background: rgba(255,255,255,0.12);
    padding:20px;
    border-radius:15px;
    color:white;
    text-align:center;
}

/* USER MESSAGE */
.user-msg {
    background:#4CAF50;
    padding:12px;
    border-radius:12px;
    color:white;
    margin-bottom:10px;
}

/* BOT MESSAGE */
.bot-msg {
    background: rgba(255,255,255,0.08);
    padding:20px;
    border-radius:12px;
    color:white;
    font-size:15px;
    line-height:1.6;
    margin-bottom:10px;
    max-height:300px;
    overflow-y:auto;
}

/* General text */
body {
    color: white !important;
}

/* App text */
.stMarkdown, .stText, p {
    color: white !important;
}

/* FLOATING CHAT INPUT */
.stChatInputContainer {
    position: fixed;
    bottom: 10px;
    left: 0;
    right: 0;
    padding: 10px 20px;
    background: transparent !important;
}

/* INPUT BOX STYLE */
.stChatInputContainer > div {
    background: rgba(0,0,0,0.3) !important;
    border-radius: 12px;
}

/* ADD SPACE FOR INPUT */
[data-testid="stAppViewContainer"] {
    padding-bottom: 80px;
}

/* SELECTBOX FIX */
div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
    border-radius: 10px !important;
}

/* Selected dropdown value */
[data-baseweb="select"] input {
    color: black !important;
    -webkit-text-fill-color: black !important;
    font-weight: 600 !important;
}

/* Dropdown text */
[data-baseweb="select"] span {
    color: black !important;
}

/* Dropdown options */
[data-baseweb="menu"] div {
    color: black !important;
    background-color: white !important;
}

/* Dropdown arrow */
div[data-baseweb="select"] svg {
    fill: black !important;
}

/* Dropdown menu only */
div[role="listbox"] {
    background-color: white !important;
}

/* Dropdown options only */
div[role="option"] {
    color: black !important;
    background-color: white !important;
    font-size: 16px !important;
}

/* Dropdown hover */
div[role="option"]:hover {
    background-color: #f1f1f1 !important;
}


/* Labels */
.stSelectbox label {
    color: white !important;
    font-size: 18px;
    font-weight: bold;
}
            
/* Sidebar title */
section[data-testid="stSidebar"] h1 {
    color: white !important;
    font-size: 40px !important;
    font-weight: bold !important;
}

/* Sidebar headings */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: white !important;
}

/* Sidebar labels */
section[data-testid="stSidebar"] label {
    color: white !important;
    font-weight: bold !important;
}

/* Sidebar markdown text */
section[data-testid="stSidebar"] .stMarkdown {
    color: white !important;
}

/* New Chat button */
.stButton > button {
    background-color: white !important;
    color: black !important;
    border-radius: 12px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border: none !important;
    padding: 10px 20px !important;
    width: 100%;
}

/* Force button text visible */
.stButton > button p {
    color: black !important;
}

            
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🏦 Banking AI")
    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("⚙ Settings")

    language = st.selectbox("🌐 Language", ["English", "Hindi", "Telugu"])
    service = st.selectbox("Service", [
        "Savings", "Loans", "Credit Card", "KYC", "Complaints"
    ])

# ---------------- TITLE ----------------
st.markdown("<div class='title'>Banking Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask questions from RBI documents</div>", unsafe_allow_html=True)

# ---------------- LOAD RAG ----------------
@st.cache_resource
def load_rag():
    return initialize_rag_system()

db = load_rag()

# ---------------- CARDS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card'>🏦 Accounts</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>💳 Cards</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>📜 RBI Rules</div>", unsafe_allow_html=True)

# ---------------- CHAT HISTORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>👤 {msg['content']}</div>", unsafe_allow_html=True)

    else:
        with st.container():
            st.markdown(
            """
            <h2 style='color:white;'>
            🤖 Answer
            </h2>
            """,
            unsafe_allow_html=True
        )

        # Styled answer box
        st.markdown(
            """
            <style>
            .answer-box {
                background: rgba(255,255,255,0.08);
                padding: 20px;
                border-radius: 12px;
                color: white;
                font-size: 17px;
                line-height: 1.8;
                margin-bottom: 15px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='answer-box'>{msg['content']}</div>",
            unsafe_allow_html=True
        )


    
# ---------------- INPUT ----------------
query = st.chat_input("Ask your banking question...")

if query:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # ---------------- TRANSLATE QUESTION ----------------
    if language == "Hindi":
        query_en = GoogleTranslator(
            source='hi',
            target='en'
        ).translate(query)

    elif language == "Telugu":
        query_en = GoogleTranslator(
            source='te',
            target='en'
        ).translate(query)
    else:
        query_en = query

    # ---------------- GET ANSWER ----------------
    with st.spinner("Thinking... 🤖"):

        answer = search_bank_answer(db, query_en)

        # ---------------- TRANSLATE ANSWER ----------------
        if language == "Hindi":

            answer = translator.translate(
                answer,
                src="en",
                dest="hi"
            ).text

        elif language == "Telugu":

            answer = translator.translate(
                answer,
                src="en",
                dest="te"
            ).text

    # Save assistant answer
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()