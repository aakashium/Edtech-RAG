import streamlit as st 
import os
import sys 
import json 
import nest_asyncio

# import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine import get_router_engine

# Apply nest_asyncio for LlamaIndex within Streamlit
nest_asyncio.apply()

# Configuration
st.set_page_config(
    page_title="University RAG Tutor",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    </style>
""", unsafe_allow_html=True)

# Initialization

@st.cache_resource(show_spinner="Loading Knowledge Base...")
def initialize_engine():
    """
    Cache the engine so we don't reload ChromaDB/LLM on every interaction.
    """
    return get_router_engine()

def parse_json_quiz(json_text):
    """
    Attempts to parse JSON output from the Examiner tool.
    Returns python object or None if failed.
    """
    try:
        clean_text = json_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text.split("```json")[1].split("```")[0]
        elif clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1].split("```")[0]
            
        return json.loads(clean_text)
    except Exception:
        return None

# Load the engine
if "engine" not in st.session_state:
    st.session_state.engine = initialize_engine()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI Tutor. Ask me to explain a concept, summarize a lecture, or generate a quiz!"}
    ]

# Sidebar
with st.sidebar:
    st.title("🎓 Course Navigator")
    st.info(
        "**Modes:**\n"
        "1. **Tutor:** Ask general questions.\n"
        "2. **Summarizer:** Ask for revision notes.\n"
        "3. **Examiner:** Ask for a 'quiz' or 'test'."
    )
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface

st.title("University Course Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], list) and message.get("is_quiz", False):
             # Render Quiz UI for history
             for i, q in enumerate(message["content"]):
                st.markdown(f"**Q{i+1}: {q['question']}**")
                st.markdown(f"*Answer:* ||{q['answer']}||") 
        else:
            st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your slides..."):
    # Add User Message to History
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Query the Router Engine
                response = st.session_state.engine.query(prompt)
                response_text = str(response)
                
                # Detect Output Type
                quiz_data = parse_json_quiz(response_text)
                
                if quiz_data and isinstance(quiz_data, list):
                    # Quiz Rendering Mode
                    st.success("Quiz Generated!")
                    for i, q in enumerate(quiz_data):
                        with st.expander(f"Question {i+1}: {q.get('question', 'Unknown')}", expanded=True):
                            # Display options as radio buttons 
                            options = q.get('options', [])
                            st.radio(f"Select answer for Q{i+1}", options, key=f"q_{i}_{len(st.session_state.messages)}")
                            
                            # Reveal Answer Button
                            if st.button(f"Show Answer Q{i+1}", key=f"btn_{i}_{len(st.session_state.messages)}"):
                                st.markdown(f"**Correct Answer:** {q.get('answer')}")
                                st.info(f"**Explanation:** {q.get('explanation')}")
                    
                    # Save to history with a flag
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": quiz_data, 
                        "is_quiz": True
                    })

                else:
                    # Standard Text Mode
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                # Source Attribution 
                if response.source_nodes:
                    with st.expander("🔍 View Source Material"):
                        for node in response.source_nodes:
                            # Extract score and content
                            score = f"{node.score:.2f}" if node.score else "N/A"
                            # Truncate content for display
                            content_preview = node.node.get_content()[:300] + "..."
                            
                            st.markdown(f"""
                            <div class="source-box">
                                <b>Relevance Score: {score}</b><br>
                                {content_preview}
                            </div>
                            """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")