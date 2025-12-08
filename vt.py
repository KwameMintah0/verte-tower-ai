import streamlit as st
import os
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PIL import Image

# --- 1. PAGE CONFIGURATION (The "App" Look) ---
st.set_page_config(
    page_title="Verte Tower OS",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to hide standard Streamlit clutter
st.markdown("""
    <style>
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. HEADER / BRANDING ---
# You can replace this URL with a photo of your actual Verte Tower later
st.image("https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?q=80&w=2070&auto=format&fit=crop", height=150, use_column_width=True)
st.title("🌱 Verte Tower Control Center")
st.markdown("### AI Agronomist & Diagnostic System")

# --- 3. SIDEBAR SETUP ---
with st.sidebar:
    st.header("⚙️ System Control")
    
    # API Key Logic
    if "GOOGLE_API_KEY" in st.secrets:
        st.success("✅ System Online (Key Loaded)")
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = st.text_input("🔑 Enter Google API Key", type="password")
        if not api_key:
            st.warning("⚠️ System Offline. Key required.")

    st.markdown("---")
    st.header("📚 Knowledge Link")
    uploaded_files = st.file_uploader("Upload Manuals", accept_multiple_files=True, type=['pdf'])
    
    if st.button("🔄 Sync/Train AI", type="primary"):
        st.session_state.train_trigger = True
    
    st.markdown("---")
    st.info("ℹ️ **Verte Tower Specs:**\n- HPA System\n- Solar Powered\n- 50L Reservoir")

# --- 4. BACKEND FUNCTIONS ---
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

# --- 5. SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to the Verte Tower OS. All systems nominal. How can I assist?"}]
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "chunk_metadatas" not in st.session_state:
    st.session_state.chunk_metadatas = []
if "train_trigger" not in st.session_state:
    st.session_state.train_trigger = False

# --- 6. MAIN APP LOGIC ---
if api_key:
    genai.configure(api_key=api_key)
    
    # NAVIGATION TABS
    tab1, tab2 = st.tabs(["💬 Operations Chat", "🔬 Plant Pathology Lab"])

    # --- TAB 1: CHAT INTERFACE ---
    with tab1:
        @st.cache_resource
        def load_embedding_model():
            return SentenceTransformer('all-MiniLM-L6-v2')
        
        try:
            embed_model = load_embedding_model()
        except Exception as e:
            st.error(f"Error loading core systems: {e}")

        # Training Trigger
        if uploaded_files and st.session_state.train_trigger:
            with st.spinner("🔄 Integrating new data into Neural Network..."):
                try:
                    chunks, metadatas = get_pdf_data(uploaded_files)
                    st.session_state.text_chunks = chunks
                    st.session_state.chunk_metadatas = metadatas
                    embeddings = embed_model.encode(chunks)
                    dimension = embeddings.shape[1]
                    index = faiss.IndexFlatL2(dimension)
                    index.add(np.array(embeddings))
                    st.session_state.vector_index = index
                    st.session_state.train_trigger = False # Reset trigger
                    st.toast("✅ Knowledge Base Updated Successfully!", icon="💾")
                except Exception as e:
                    st.error(f"Training Failed: {e}")

        # Chat Display
        for msg in st.session_state.messages:
            # Custom Avatars
            avatar = "🧑‍🌾" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.write(msg["content"])
                if "sources" in msg:
                    with st.expander("🔍 Verified Sources"):
                        for src in msg["sources"]:
                            st.caption(f"📄 **{src['source']}** (Page {src['page']})")
                            st.text(src['text'][:150] + "...")

        # Chat Input
        if prompt := st.chat_input("Enter command or question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user", avatar="🧑‍🌾").write(prompt)

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

                    # AI Model Logic (Gemini Pro)
                    model = genai.GenerativeModel('gemini-pro')
                    full_prompt = f"""Role: Chief Systems Agronomist for 'Verte Tower'.
                    System: High-Pressure Aeroponics (HPA), Solar, U-Shaped Vertical Tower.
                    Data: {relevant_text}
                    User Query: {prompt}"""
                    
                    response = model.generate_content(full_prompt)
                    
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(response.text)
                        with st.expander("🔍 Verified Sources"):
                            for src in sources:
                                st.caption(f"📄 **{src['source']}** (Page {src['page']})")
                                st.text(src['text'][:150] + "...")
                    
                    st.session_state.messages.append({"role": "assistant", "content": response.text, "sources": sources})
                except Exception as e:
                    st.error(f"System Error: {e}")
            else:
                st.warning("⚠️ Knowledge Base Empty. Please upload manuals and click 'Sync/Train'.")

    # --- TAB 2: VISUAL DIAGNOSTICS (Split Layout) ---
    with tab2:
        st.subheader("📸 Visual Symptom Analysis")
        
        col1, col2 = st.columns([1, 2]) # Split screen: 1/3 Image, 2/3 Results
        
        with col1:
            img_file = st.file_uploader("Upload Plant Photo", type=["jpg", "png", "jpeg"])
            if img_file:
                image = Image.open(img_file)
                st.image(image, caption="Specimen", use_container_width=True)
                analyze_btn = st.button("🔬 Run Analysis", type="primary")
            
        with col2:
            if img_file and analyze_btn:
                with st.spinner("🔬 Analyzing leaf tissue patterns..."):
                    try:
                        # AI Vision Logic (Gemini Pro Vision)
                        vision_model = genai.GenerativeModel('gemini-pro-vision')
                        vision_prompt = """
                        Analyze this plant image for the Verte Tower Aeroponic system.
                        1. ID Crop.
                        2. Detect Symptoms (Chlorosis, Necrosis, Wilting).
                        3. Diagnose Issue (Deficiency, Pest, Pathogen).
                        4. Rx: Provide organic treatment.
                        """
                        response = vision_model.generate_content([vision_prompt, image])
                        
                        st.success("Analysis Complete")
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Analysis Failed: {e}")
            elif not img_file:
                st.info("👈 Upload an image to begin analysis.")
            else:
                st.info("👆 Click 'Run Analysis' to process the image.")

