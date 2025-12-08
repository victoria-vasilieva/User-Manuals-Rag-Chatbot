import streamlit as st
from src.model_loader import initialise_llm, get_embedding_model
from src.engine import get_chat_engine

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Roborock RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ---------- HEADER ----------
st.image("images/roborock_logo.jpg", width=260)
st.markdown("## Roborock Manuals Assistant")
st.caption("Unofficial assistant for portfolio/demo use")
st.markdown("---")

# ---------- SESSION ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_engine" not in st.session_state:
    llm = initialise_llm()
    embed_model = get_embedding_model()
    st.session_state.chat_engine = get_chat_engine(llm, embed_model)

# ---------- MODE SELECT ----------
mode = st.radio(
    "What kind of question is this?",
    ["General question", "Specific model"],
    horizontal=True
)

# ---------- MODEL SELECT (only if specific) ----------
model = None
if mode == "Specific model":
    model = st.selectbox(
        "Select Roborock model",
        [
            "Roborock S7 Pro Ultra",
            "Roborock S8 MaxV Ultra",
            "Roborock Qrevo Pro"
        ]
    )

st.markdown("---")

# ---------- INPUT + PROCESS ----------
def send_message():
    user_input = st.session_state.user_input.strip()
    if not user_input:
        return

    if mode == "Specific model" and model:
        prompt = f"[MODEL: {model}] {user_input}"
    else:
        prompt = user_input

    response = st.session_state.chat_engine.chat(prompt).response

    # Show newest messages first
    st.session_state.chat_history.insert(0, ("Assistant", response))
    st.session_state.chat_history.insert(0, ("You", user_input))

    # Clear input field
    st.session_state.user_input = ""

st.text_input("Ask your question:", key="user_input", on_change=send_message)

# ---------- CHAT DISPLAY ----------
st.subheader("Conversation")

if not st.session_state.chat_history:
    st.info("Ask something to start the chat.")

for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧍 You:** {msg}")
    else:
        st.markdown(f"**🤖 Assistant:** {msg}")

# ---------- FOOTER ----------
st.markdown("---")
st.caption(
    "Roborock logo — CC BY-SA 4.0 via Wikimedia Commons | "
    "Portfolio demo project"
)
