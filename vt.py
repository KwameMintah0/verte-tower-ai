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
    st.session_state.messages = [{"role": "assistant", "content": "System Online. Ready for text, voice, or images."}]
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
    
    # --- IMAGE UPLOADER (Attachments) ---
    st.subheader("📎 Attachments")
    uploaded_img = st.file_uploader("Attach Image", type=["jpg", "png", "jpeg"], key="chat_image_upload")
    if uploaded_img:
        st.info("✅ Image attached! Type or speak below.")

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

    # --- CHAT HISTORY CONTAINER ---
    # We use a container to keep messages distinct from the input area
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if "image" in msg and msg["image"]:
                    st.image(msg["image"], width=300)
                st.write(msg["content"])

    # --- UNIFIED INPUT AREA ---
    # We place the Audio Input here so it sits right above the fixed chat bar
    # This is the closest we can get to "Inside the box" in Streamlit
    audio_value = st.audio_input("🎤 Voice Note (Click to record)")
    
    # Standard Chat Input (Always pinned to bottom)
    prompt = st.chat_input("Type your message...")

    # --- INPUT PROCESSING ---
    final_prompt = None
    
    # 1. Check Voice First
    if audio_value:
        model = genai.GenerativeModel("gemini-2.5-flash")
        with st.spinner("Transcribing..."):
            # We treat the transcription as the prompt
            response = model.generate_content(["Transcribe this audio exactly.", audio_value])
            final_prompt = response.text 
    
    # 2. Check Text Input (Overwrites voice if both exist in same tick, though unlikely)
    if prompt:
        final_prompt = prompt

    # --- EXECUTE IF WE HAVE INPUT ---
    if final_prompt:
        # Prepare User Payload
        user_msg = {"role": "user", "content": final_prompt}
        
        # Check Image Attachment
        img_data = None
        if uploaded_img:
            img_data = Image.open(uploaded_img)
            user_msg["image"] = img_data
        
        # Update UI & History
        with chat_container:
            with st.chat_message("user"):
                if img_data:
                    st.image(img_data, width=300)
                st.write(final_prompt)
        st.session_state.messages.append(user_msg)

        # Build AI Context
        model = genai.GenerativeModel('gemini-2.5-flash')
        content_parts = []
        
        # Text Logic
        if chat_mode == "📚 Verte Manuals (PDF)" and not img_data:
            if st.session_state.vector_index:
                query_embedding = embed_model.encode([final_prompt])
                D, I = st.session_state.vector_index.search(np.array(query_embedding), k=3)
                relevant_text = ""
                for idx in I[0]:
                    if idx < len(st.session_state.text_chunks):
                        relevant_text += st.session_state.text_chunks[idx] + "\n"
                full_prompt = f"You are a technical assistant. Use ONLY the Context below. \nContext: {relevant_text} \nQuestion: {final_prompt}"
                content_parts.append(full_prompt)
            else:
                st.warning("⚠️ Manual Mode: Please load knowledge base first.")
                content_parts.append(final_prompt)
        else:
            full_prompt = f"You are Verte AI, a specialized agricultural consultant. Answer clearly. Question: {final_prompt}"
            content_parts.append(full_prompt)

        # Image Logic
        if img_data:
            content_parts.append(img_data)

        # Generate Response
        if content_parts:
            try:
                with chat_container:
                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        full_response = ""
                        stream = model.generate_content(content_parts, stream=True)
                        
                        # Streaming Loop
                        for chunk in stream:
                            if chunk.text:
                                full_response += chunk.text
                                placeholder.markdown(full_response + "▌")
                        placeholder.markdown(full_response)
                        
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error: {e}")
