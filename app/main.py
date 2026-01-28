import streamlit as st
import os
import sys
import json
import shutil
import nest_asyncio

# Add the project root to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine import get_router_engine
# Import the ingestion function directly
from src.ingestion import run_ingestion 

# Apply nest_asyncio
nest_asyncio.apply()

# Configuration
st.set_page_config(
    page_title="University RAG Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    .source-box {
        background-color: #f0f2f6;
        border-left: 5px solid #4CAF50;
        padding: 10px;
        margin-top: 10px;
        font-size: 0.9em;
    }
    /* Style for sidebar buttons to look more actionable */
    .stButton button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Directories
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialization

@st.cache_resource(show_spinner="Loading Knowledge Base...")
def get_engine():
    """
    Cached engine loader. We can clear this cache when new data is uploaded.
    """
    return get_router_engine()

def parse_json_quiz(json_text):
    try:
        clean_text = json_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text.split("```json")[1].split("```")[0]
        elif clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1].split("```")[0]
        return json.loads(clean_text)
    except Exception:
        return None

def process_upload(uploaded_files):
    """
    Saves uploaded files to /data and triggers ingestion.
    """
    if not uploaded_files:
        return
    
    # Clear old data 
    for f in os.listdir(DATA_DIR):
        os.remove(os.path.join(DATA_DIR, f))

    # Save new files
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Trigger Ingestion Pipeline
    with st.spinner("Reading slides & updating knowledge base... (This may take a minute)"):
        try:
            run_ingestion()
            
            # CLEAR CACHE so the new index is loaded
            st.cache_resource.clear()
            st.success("Ingestion Complete! You can now chat with your slides.")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload your lecture slides in the sidebar, then ask me to explain concepts or create quizzes!"}
    ]

# Sidebar
with st.sidebar:
    st.title("Course Navigator")
    
    # Upload Section
    st.subheader("1. Upload Slides")
    uploaded_files = st.file_uploader(
        "Upload PDF Lectures", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process & Ingest Data", type="primary"):
            process_upload(uploaded_files)
            # Reload the engine immediately
            st.session_state.engine = get_engine()

    st.divider()

    # Quick Actions
    st.subheader("2. Quick Actions")
    
    # Define callback functions for buttons
    def set_prompt(text):
        st.session_state.user_prompt = text

    col1, col2 = st.columns(2)
    with col1:
        st.button("Summarize", on_click=set_prompt, args=("Create a detailed bullet-point summary of the uploaded slides.",))
    with col2:
        st.button("Create Quiz", on_click=set_prompt, args=("Generate a 3-question multiple choice quiz on this topic.",))
    
    st.button("Explain Key Concepts", on_click=set_prompt, args=("Explain the top 3 most difficult concepts in these slides like I'm a beginner.",))

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main

# Check if we have a prompt from the sidebar buttons or the text input
user_input = None

# Check Sidebar Input
if "user_prompt" in st.session_state and st.session_state.user_prompt:
    user_input = st.session_state.user_prompt
    # Clear it so it doesn't persist forever
    del st.session_state.user_prompt

# Check Text Input 
chat_input = st.chat_input("Ask a question about your slides...")
if chat_input:
    user_input = chat_input

# Chat Processing
if user_input:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": user_input})

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], list) and message.get("is_quiz", False):
             for i, q in enumerate(message["content"]):
                st.markdown(f"**Q{i+1}: {q['question']}**")
                st.markdown(f"*Answer:* ||{q['answer']}||") 
        else:
            st.markdown(message["content"])

# Generate Response 
if user_input and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Ensure engine is loaded
                engine = get_engine()
                
                response = engine.query(user_input)
                response_text = str(response)
                
                # Detect Quiz JSON
                quiz_data = parse_json_quiz(response_text)
                
                if quiz_data and isinstance(quiz_data, list):
                    st.success("Quiz Generated!")
                    for i, q in enumerate(quiz_data):
                        with st.expander(f"Question {i+1}: {q.get('question', 'Unknown')}", expanded=True):
                            options = q.get('options', [])
                            st.radio(f"Select answer for Q{i+1}", options, key=f"q_{i}_{len(st.session_state.messages)}")
                            if st.button(f"Show Answer Q{i+1}", key=f"btn_{i}_{len(st.session_state.messages)}"):
                                st.markdown(f"**Correct Answer:** {q.get('answer')}")
                                st.info(f"**Explanation:** {q.get('explanation')}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": quiz_data, 
                        "is_quiz": True
                    })

                else:
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                # Attribution
                if response.source_nodes:
                    with st.expander("View Source Material"):
                        for node in response.source_nodes:
                            score = f"{node.score:.2f}" if node.score else "N/A"
                            content_preview = node.node.get_content()[:300] + "..."
                            st.markdown(f"""
                            <div class="source-box">
                                <b>Relevance Score: {score}</b><br>
                                {content_preview}
                            </div>
                            """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")