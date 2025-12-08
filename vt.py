import streamlit as st
import os
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Verte Tower OS", page_icon="🌱", layout="wide")

# --- HEADER ---
st.image("https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?q=80&w=2070&h=500&auto=format&fit=crop", use_column_width=True)
st.title("🌱 Verte Tower")

# --- SESSION STATE INITIALIZATION (Moved up for the clear button) ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "System Online. Ready for queries."}]
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "chunk_metadatas" not in st.session_state:
    st.session_state.chunk_metadatas = []
if "train_trigger" not in st.session_state:
    st.session_state.train_trigger = False

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ System Status")
    
    if "GOOGLE_API_KEY" in st.secrets:
        st.success("✅ Key Loaded")
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = st.text_input("🔑 API Key", type="password")

    # --- DIAGNOSTIC TOOL: LIST AVAILABLE MODELS ---
    if api_key:
        genai.configure(api_key=api_key)
        with st.expander("🛠️ View Available Models"):
            try:
                # This lists what your server can ACTUALLY see
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        st.code(m.name)
            except Exception as e:
                st.error(f"List Error: {e}")

    st.header("📚 Knowledge Base")
    uploaded_files = st.file_uploader("Upload Manuals", accept_multiple_files=True, type=['pdf'])
    if st.button("🔄 Train AI"):
        st.session_state.train_trigger = True

# --- FUNCTIONS ---
def get_pdf_data(files):
    chunks = []
    metadatas = []
    for pdf in files:
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            chunk_size = 1000
            page_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            for chunk in page_chunks:
                chunks.append(chunk)
                metadatas.append({"source": pdf.name, "page": i + 1})
    return chunks, metadatas

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "System Online. Ready for queries."}]
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "chunk_metadatas" not in st.session_state:
    st.session_state.chunk_metadatas = []
if "train_trigger" not in st.session_state:
    st.session_state.train_trigger = False

# --- MAIN LOGIC ---
if api_key:
    genai.configure(api_key=api_key)
    
    tab1, tab2 = st.tabs(["💬 Chat", "📸 Vision"])

    # --- TAB 1: CHAT ---
    with tab1:
        @st.cache_resource
        def load_embedding_model():
            return SentenceTransformer('all-MiniLM-L6-v2')
        embed_model = load_embedding_model()

        if uploaded_files and st.session_state.train_trigger:
            with st.spinner("Indexing..."):
                chunks, metadatas = get_pdf_data(uploaded_files)
                st.session_state.text_chunks = chunks
                st.session_state.chunk_metadatas = metadatas
                embeddings = embed_model.encode(chunks)
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(np.array(embeddings))
                st.session_state.vector_index = index
                st.session_state.train_trigger = False
                st.success("Updated!")

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Query..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            if st.session_state.vector_index:
                query_embedding = embed_model.encode([prompt])
                D, I = st.session_state.vector_index.search(np.array(query_embedding), k=3)
                relevant_text = ""
                for idx in I[0]:
                    if idx < len(st.session_state.text_chunks):
                        relevant_text += st.session_state.text_chunks[idx] + "\n"

                # --- UPDATED MODEL CALL (Fixing 404 Error) ---
                # Using gemini-2.5-flash which is current and faster
                model = genai.GenerativeModel('gemini-2.5-flash')
                full_prompt = f"Context: {relevant_text} \n Question: {prompt}"
                try:
                    response = model.generate_content(full_prompt)
                    st.chat_message("assistant").write(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Upload manuals and train first.")

    # --- TAB 2: VISION ---
    with tab2:
        img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if img_file and st.button("Analyze"):
            image = Image.open(img_file)
            st.image(image, use_column_width=True)
            try:
                # --- UPDATED VISION CALL ---
                # The modern models are multimodal (Text + Image in one model)
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(["Diagnose this plant based on visual cues.", image])
                st.write(response.text)
            except Exception as e:
                st.error(f"Vision Error: {e}")

