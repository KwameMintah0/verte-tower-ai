import streamlit as st
import os
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PIL import Image
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Verte Tower OS", page_icon="🌱", layout="wide")

# --- HEADER ---
st.image("https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?q=80&w=2070&h=500&auto=format&fit=crop", use_column_width=True)
st.title("🌱 Verte Tower Control Center")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "System Online. I am listening."}]
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "train_trigger" not in st.session_state:
    st.session_state.train_trigger = False

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    if "GOOGLE_API_KEY" in st.secrets:
        st.success("✅ Key Loaded (Secret)")
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = st.text_input("🔑 API Key", type="password")

    st.divider()
    
    st.subheader("🤖 AI Personality")
    chat_mode = st.radio(
        "Source of Truth:",
        ["📚 Verte Manuals (PDF)", "🌍 Global Agri-Expert"],
        captions=["Strictly uses your uploads", "Knows all Agriculture/Botany"]
    )

    st.divider()
    if st.button("🗑️ Clear Chat History", type="primary"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat cleared."}]
        st.rerun()

    with st.expander("📚 Knowledge Base Management", expanded=(chat_mode == "📚 Verte Manuals (PDF)")):
        index_exists = os.path.exists("verte_index.faiss") and os.path.exists("verte_chunks.pkl")
        if st.session_state.vector_index is not None:
            st.success("Brain: ACTIVE")
        elif index_exists:
            st.info("Saved Brain Found")
            if st.button("📂 Load Saved Knowledge"):
                st.session_state.load_trigger = True
                st.rerun()
        else:
            st.warning("Brain: EMPTY")
        uploaded_files = st.file_uploader("Upload Manuals", accept_multiple_files=True, type=['pdf'])
        if st.button("🔄 Train & Save"):
            st.session_state.train_trigger = True

# --- FUNCTIONS ---
def get_pdf_data(files):
    chunks = []
    for pdf in files:
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            chunk_size = 1000
            page_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            for chunk in page_chunks:
                chunks.append(chunk)
    return chunks

# --- MAIN LOGIC ---
if api_key:
    genai.configure(api_key=api_key)
    
    tab1, tab2 = st.tabs(["💬 Chat", "📸 Vision"])

    # === TAB 1: CHAT ===
    with tab1:
        @st.cache_resource
        def load_embedding_model():
            return SentenceTransformer('all-MiniLM-L6-v2')
        embed_model = load_embedding_model()

        # Load/Train Logic
        if "load_trigger" in st.session_state and st.session_state.load_trigger:
            try:
                index = faiss.read_index("verte_index.faiss")
                with open("verte_chunks.pkl", "rb") as f:
                    chunks = pickle.load(f)
                st.session_state.vector_index = index
                st.session_state.text_chunks = chunks
                st.session_state.load_trigger = False
                st.success("Knowledge Loaded!")
            except Exception as e:
                st.error(f"Load Error: {e}")

        if uploaded_files and st.session_state.train_trigger:
            with st.spinner("Indexing..."):
                chunks = get_pdf_data(uploaded_files)
                st.session_state.text_chunks = chunks
                embeddings = embed_model.encode(chunks)
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(np.array(embeddings))
                st.session_state.vector_index = index
                faiss.write_index(index, "verte_index.faiss")
                with open("verte_chunks.pkl", "wb") as f:
                    pickle.dump(chunks, f)
                st.session_state.train_trigger = False
                st.success("Brain Updated!")

        # --- DISPLAY CHAT HISTORY (Above input) ---
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

        # --- INPUT AREA (Below history) ---
        prompt = None
        
        # A. Audio Input
        audio_value = st.audio_input("🎤 Record Voice Note")
        if audio_value:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(["Transcribe this audio exactly.", audio_value])
            prompt = response.text 

        # B. Text Input
        if not prompt:
            prompt = st.chat_input("Ask Verte Tower...")

        # --- PROCESSING LOGIC ---
        if prompt:
            # Display user message
            with chat_container:
                st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Setup Model
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Context Logic
            if chat_mode == "📚 Verte Manuals (PDF)":
                if st.session_state.vector_index:
                    query_embedding = embed_model.encode([prompt])
                    D, I = st.session_state.vector_index.search(np.array(query_embedding), k=3)
                    relevant_text = ""
                    for idx in I[0]:
                        if idx < len(st.session_state.text_chunks):
                            relevant_text += st.session_state.text_chunks[idx] + "\n"
                    full_prompt = f"You are a technical assistant. Use ONLY the Context below. \nContext: {relevant_text} \nQuestion: {prompt}"
                else:
                    full_prompt = None
                    st.warning("⚠️ Manual Mode: Please load knowledge base first.")
            else:
                full_prompt = (
                    f"You are Verte AI, a specialized agricultural consultant. "
                    f"Answer this clearly and concisely. Question: {prompt}"
                )

            # Generate & Stream Response
            if full_prompt:
                try:
                    with chat_container:
                        with st.chat_message("assistant"):
                            stream = model.generate_content(full_prompt, stream=True)
                            response_text = st.write_stream(stream)
                            
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error(f"Error: {e}")

    # === TAB 2: VISION ===
    with tab2:
        img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if img_file and st.button("Analyze"):
            image = Image.open(img_file)
            st.image(image, use_column_width=True)
            with st.spinner("Analyzing plant health..."):
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    response = model.generate_content(["Diagnose this plant based on visual cues. Keep it concise.", image])
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Vision Error: {e}")
