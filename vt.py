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
    st.session_state.messages = [{"role": "assistant", "content": "System Online. Ready for text or images."}]
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
    
    # --- NEW: IMAGE UPLOADER IN SIDEBAR (Acts as Attachment) ---
    st.subheader("📎 Attachments")
    uploaded_img = st.file_uploader("Attach Image to Chat", type=["jpg", "png", "jpeg"], key="chat_image_upload")
    if uploaded_img:
        st.info("✅ Image attached! Type your question below.")

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

    # --- DISPLAY CHAT HISTORY ---
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                # If the message has an image, show it first
                if "image" in msg and msg["image"]:
                    st.image(msg["image"], width=300)
                # Show the text
                st.write(msg["content"])

    # --- INPUT AREA ---
    prompt = None
    
    # Audio Input
    audio_value = st.audio_input("🎤 Record Voice Note")
    if audio_value:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(["Transcribe this audio exactly.", audio_value])
        prompt = response.text 

    # Text Input
    if not prompt:
        prompt = st.chat_input("Ask Verte Tower (or upload an image first)...")

    # --- PROCESSING LOGIC ---
    if prompt:
        # Prepare the User Message Payload
        user_msg = {"role": "user", "content": prompt}
        
        # Check for attached image
        img_data = None
        if uploaded_img:
            img_data = Image.open(uploaded_img)
            user_msg["image"] = img_data # Add image to history
        
        # Display User Message
        with chat_container:
            with st.chat_message("user"):
                if img_data:
                    st.image(img_data, width=300)
                st.write(prompt)
        
        st.session_state.messages.append(user_msg)

        # --- PREPARE AI INPUT ---
        model = genai.GenerativeModel('gemini-2.5-flash')
        content_parts = []
        
        # 1. Add Text Prompt
        if chat_mode == "📚 Verte Manuals (PDF)" and not img_data:
            # RAG Logic (Only works for text-only queries generally)
            if st.session_state.vector_index:
                query_embedding = embed_model.encode([prompt])
                D, I = st.session_state.vector_index.search(np.array(query_embedding), k=3)
                relevant_text = ""
                for idx in I[0]:
                    if idx < len(st.session_state.text_chunks):
                        relevant_text += st.session_state.text_chunks[idx] + "\n"
                full_prompt = f"You are a technical assistant. Use ONLY the Context below. \nContext: {relevant_text} \nQuestion: {prompt}"
                content_parts.append(full_prompt)
            else:
                st.warning("⚠️ Manual Mode: Please load knowledge base first.")
                content_parts.append(prompt)
        else:
            # General Logic (Works for Text OR Text+Image)
            full_prompt = (
                f"You are Verte AI, a specialized agricultural consultant. "
                f"Answer this clearly. Question: {prompt}"
            )
            content_parts.append(full_prompt)

        # 2. Add Image (if exists)
        if img_data:
            content_parts.append(img_data)

        # --- GENERATE RESPONSE ---
        if content_parts:
            try:
                with chat_container:
                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        full_response = ""
                        
                        # Stream the response
                        stream = model.generate_content(content_parts, stream=True)
                        
                        for chunk in stream:
                            if chunk.text:
                                full_response += chunk.text
                                placeholder.markdown(full_response + "▌")
                        
                        placeholder.markdown(full_response)
                        
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error: {e}")
