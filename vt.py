# Install the stable basics (No LangChain)
!pip install -q streamlit google-generativeai faiss-cpu sentence-transformers pypdf

# Install the tunnel tool (Cloudflare)
!wget -q -O cloudflared-linux-amd64 https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64

%%writefile app.py
import streamlit as st
import os
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Verte Tower AI", page_icon="🌿", layout="wide")
st.title("🌿 Verte Tower AI")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Configuration")
    api_key = st.text_input("Google API Key", type="password")
    
    st.header("2. Knowledge Base")
    uploaded_files = st.file_uploader("Upload Manuals (PDF)", accept_multiple_files=True, type=['pdf'])
    train_btn = st.button("Train AI on Manuals")
    
    st.markdown("---")
    st.info("💡 **Tip:** Always click 'Train' after uploading new files or restarting the app.")

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
    # --- UPDATED GREETING HERE ---
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your Verte Tower today?"}]

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "chunk_metadatas" not in st.session_state:
    st.session_state.chunk_metadatas = []

# --- MAIN LOGIC ---
if api_key:
    genai.configure(api_key=api_key)
    
    @st.cache_resource
    def load_embedding_model():
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    try:
        embed_model = load_embedding_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")

    # TRAINING PHASE
    if uploaded_files and train_btn:
        with st.spinner("Analyzing Manuals & Indexing..."):
            try:
                # 1. Process PDF with Metadata
                chunks, metadatas = get_pdf_data(uploaded_files)
                st.session_state.text_chunks = chunks
                st.session_state.chunk_metadatas = metadatas
                
                # 2. Embed
                embeddings = embed_model.encode(chunks)
                
                # 3. Build Index
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(np.array(embeddings))
                st.session_state.vector_index = index
                
                st.success(f"✅ Indexed {len(chunks)} sections from {len(uploaded_files)} manuals.")
            except Exception as e:
                st.error(f"Training Failed: {e}")

    # CHAT HISTORY
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg:
                with st.expander("📚 View Sources"):
                    for src in msg["sources"]:
                        st.caption(f"**File:** {src['source']} | **Page:** {src['page']}")
                        st.text(src['text'][:150] + "...")

    # CHAT INPUT
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # SAFETY CHECK: Ensure we have trained data
        if st.session_state.vector_index is not None:
            try:
                # 1. Search
                query_embedding = embed_model.encode([prompt])
                D, I = st.session_state.vector_index.search(np.array(query_embedding), k=3)
                
                # 2. Retrieve Data & Sources (WITH SAFETY CHECK)
                indices = I[0]
                relevant_text = ""
                sources = []
                
                for idx in indices:
                    # Check if index exists to prevent crashes
                    if idx < len(st.session_state.text_chunks) and idx < len(st.session_state.chunk_metadatas):
                        chunk_text = st.session_state.text_chunks[idx]
                        metadata = st.session_state.chunk_metadatas[idx]
                        
                        relevant_text += f"\n--- Source: {metadata['source']} (Page {metadata['page']}) ---\n{chunk_text}\n"
                        sources.append({"source": metadata['source'], "page": metadata['page'], "text": chunk_text})

                # 3. Generate Answer
                if not relevant_text:
                    st.warning("I couldn't find relevant info in the documents.")
                else:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    full_prompt = f"""You are the Chief Agronomist for the 'Verte Tower' project.
                    
                    PROJECT CONTEXT:
                    - System: U-shaped vertical aeroponic tower (HPA).
                    - Power: Solar + Battery.
                    - Mobility: Wheeled stand.
                    
                    INSTRUCTIONS:
                    Answer the question using ONLY the context provided below.
                    Cite the specific manual name if mentioned.
                    
                    CONTEXT FROM MANUALS:
                    {relevant_text}
                    
                    QUESTION:
                    {prompt}
                    """
                    
                    response = model.generate_content(full_prompt)
                    
                    # 4. Display & Save
                    with st.chat_message("assistant"):
                        st.write(response.text)
                        with st.expander("📚 View Sources"):
                            for src in sources:
                                st.caption(f"**File:** {src['source']} | **Page:** {src['page']}")
                                st.text(src['text'][:150] + "...")

                    st.session_state.messages.append({"role": "assistant", "content": response.text, "sources": sources})
            
            except Exception as e:
                st.error(f"An error occurred during generation: {e}. Please try clicking 'Train AI' again.")
            
        else:
            st.error("⚠️ System not trained. Please upload manuals and click 'Train AI on Manuals' in the sidebar.")

import subprocess
import time

# Run Streamlit
subprocess.Popen(["streamlit", "run", "app.py"])

# Start Tunnel
print("Starting Tunnel... please wait 10 seconds...")
with open('tunnel.log', 'w') as f:
    subprocess.Popen(["./cloudflared-linux-amd64", "tunnel", "--url", "http://localhost:8501"], stdout=f, stderr=f)

time.sleep(10)

# Get Link
!grep -o 'https://.*\.trycloudflare.com' tunnel.log | head -n 1 | xargs echo "CLICK THIS LINK -->"