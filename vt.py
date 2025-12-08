import streamlit as st
import os
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Verte Tower AI", page_icon="🌿", layout="wide")
st.title("🌿 Verte Tower AI")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Configuration")
    if "GOOGLE_API_KEY" in st.secrets:
        st.success("✅ API Key loaded securely")
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = st.text_input("Google API Key", type="password")

    st.header("2. Knowledge Base")
    uploaded_files = st.file_uploader("Upload Manuals (PDF)", accept_multiple_files=True, type=['pdf'])
    train_btn = st.button("Train AI on Manuals")
    
    st.info("💡 **New:** Go to the 'Plant Doctor' tab to diagnose sick plants with photos.")

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
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I can help you with text questions or visual diagnosis. Choose a tab above!"}]
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "chunk_metadatas" not in st.session_state:
    st.session_state.chunk_metadatas = []

# --- MAIN LOGIC ---
if api_key:
    genai.configure(api_key=api_key)
    
    # SETUP TABS
    tab1, tab2 = st.tabs(["💬 Chat & Manuals", "📸 Plant Doctor (Vision)"])

    # --- TAB 1: TEXT CHAT (Uses gemini-pro) ---
    with tab1:
        @st.cache_resource
        def load_embedding_model():
            return SentenceTransformer('all-MiniLM-L6-v2')
        
        try:
            embed_model = load_embedding_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")

        # Training
        if uploaded_files and train_btn:
            with st.spinner("Analyzing Manuals & Indexing..."):
                try:
                    chunks, metadatas = get_pdf_data(uploaded_files)
                    st.session_state.text_chunks = chunks
                    st.session_state.chunk_metadatas = metadatas
                    embeddings = embed_model.encode(chunks)
                    dimension = embeddings.shape[1]
                    index = faiss.IndexFlatL2(dimension)
                    index.add(np.array(embeddings))
                    st.session_state.vector_index = index
                    st.success(f"✅ Indexed {len(chunks)} sections.")
                except Exception as e:
                    st.error(f"Training Failed: {e}")

        # Chat History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "sources" in msg:
                    with st.expander("📚 View Sources"):
                        for src in msg["sources"]:
                            st.caption(f"**File:** {src['source']} | **Page:** {src['page']}")
                            st.text(src['text'][:150] + "...")

        # Chat Input
        if prompt := st.chat_input("Ask about nutrients, pressure, or solar..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            if st.session_state.vector_index is not None:
                try:
                    query_embedding = embed_model.encode([prompt])
                    D, I = st.session_state.vector_index.search(np.array(query_embedding), k=3)
                    indices = I[0]
                    relevant_text = ""
                    sources = []
                    
                    for idx in indices:
                        if idx < len(st.session_state.text_chunks):
                            chunk_text = st.session_state.text_chunks[idx]
                            metadata = st.session_state.chunk_metadatas[idx]
                            relevant_text += f"\n--- Source: {metadata['source']} (Page {metadata['page']}) ---\n{chunk_text}\n"
                            sources.append({"source": metadata['source'], "page": metadata['page'], "text": chunk_text})

                    # --- CHANGE 1: Use 'gemini-pro' for text ---
                    model = genai.GenerativeModel('gemini-pro')
                    full_prompt = f"""You are the Agronomist for the 'Verte Tower'.
                    CONTEXT: U-shaped vertical aeroponic tower (HPA), Solar Powered.
                    MANUAL INFO: {relevant_text}
                    QUESTION: {prompt}"""
                    
                    response = model.generate_content(full_prompt)
                    
                    with st.chat_message("assistant"):
                        st.write(response.text)
                        with st.expander("📚 View Sources"):
                            for src in sources:
                                st.caption(f"**File:** {src['source']} | **Page:** {src['page']}")
                                st.text(src['text'][:150] + "...")
                    
                    st.session_state.messages.append({"role": "assistant", "content": response.text, "sources": sources})
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("⚠️ Please upload manuals and click 'Train' first.")

    # --- TAB 2: PLANT DOCTOR (Uses gemini-pro-vision) ---
    with tab2:
        st.header("📸 AI Plant Diagnosis")
        st.write("Upload a photo of your sick plant (leaves or roots). The AI will identify the issue.")
        
        img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        
        if img_file:
            image = Image.open(img_file)
            st.image(image, caption="Uploaded Plant Photo", width=300)
            
            analyze_btn = st.button("Diagnose Issue")
            
            if analyze_btn:
                with st.spinner("Analyzing leaf patterns and discoloration..."):
                    try:
                        # --- CHANGE 2: Use 'gemini-pro-vision' for images ---
                        vision_model = genai.GenerativeModel('gemini-pro-vision')
                        
                        vision_prompt = """
                        Act as an expert Plant Pathologist. Analyze this image carefully.
                        1. Identify the crop if possible.
                        2. Describe the visual symptoms (e.g., interveinal chlorosis, tip burn).
                        3. Diagnose the likely cause (Nutrient deficiency, Pest, or Disease).
                        4. Recommend an organic treatment suitable for Aeroponics.
                        """
                        
                        response = vision_model.generate_content([vision_prompt, image])
                        
                        st.success("Diagnosis Complete")
                        st.markdown(response.text)
                        
                    except Exception as e:
                        st.error(f"Vision Analysis Failed: {e}")
