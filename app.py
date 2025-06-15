import streamlit as st
from PyPDF2 import PdfReader
import base64
import os
import tempfile
from datetime import datetime

# LangChain imports - minimal and efficient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.prompts import PromptTemplate

# Configure page
st.set_page_config(
    page_title="Financial Analyst",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean chat UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 100%;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 20%;
    }
    .chat-input {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem 0;
        border-top: 1px solid #ddd;
    }
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #1f77b4;
        padding: 10px 20px;
    }
    .stButton > button {
        border-radius: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 30px;
        font-weight: bold;
    }
    .processing-indicator {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class LightweightFinancialRAG:
    def __init__(self):
        self.vectorstore = None
        self.llm = None
        self.embeddings = None
        self.is_ready = False
        
    def initialize_models(self, api_key):
        """Initialize models with minimal setup"""
        if self.is_ready:
            return True
            
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.3
            )
            
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
            
            self.is_ready = True
            return True
        except Exception as e:
            st.error(f"Model initialization failed: {str(e)}")
            return False
    
    def extract_pdf_text_fast(self, pdf_file):
        """Extract all PDF text using multiple methods for better table handling"""
        try:
            # Method 1: Try UnstructuredPDFLoader for better table handling
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Use UnstructuredPDFLoader
                loader = UnstructuredPDFLoader(tmp_file_path)
                documents = loader.load()
                
                # Combine all document content
                text = ""
                for doc in documents:
                    if doc.page_content:
                        text += doc.page_content + "\n"
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
                if text and len(text.strip()) > 100:
                    return text
                    
            except Exception as e:
                st.warning(f"UnstructuredPDFLoader failed, trying PyPDF2: {str(e)}")
            
            # Method 2: Fallback to PyPDF2 with better error handling
            try:
                pdf_reader = PdfReader(pdf_file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text + "\n"
                    except Exception as page_error:
                        st.warning(f"Error reading page {page_num + 1}: {str(page_error)}")
                        continue
                
                return text
                
            except Exception as e:
                st.error(f"PyPDF2 extraction failed: {str(e)}")
                return ""
                
        except Exception as e:
            st.error(f"PDF extraction completely failed: {str(e)}")
            return ""
    
    def process_and_store_fast(self, pdf_file):
        """Enhanced processing with better error handling and validation"""
        try:
            # Extract text with progress indication
            st.info("📄 Extracting text from PDF...")
            raw_text = self.extract_pdf_text_fast(pdf_file)
            
            if not raw_text:
                st.error("❌ No text could be extracted from PDF")
                return False
                
            if len(raw_text.strip()) < 50:
                st.error("❌ PDF contains insufficient text content")
                return False
            
            st.info(f"✅ Extracted {len(raw_text)} characters from PDF")
            
            # Create chunks with better handling
            st.info("🔄 Creating text chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            # Split text into chunks
            chunks = text_splitter.split_text(raw_text)
            
            if not chunks:
                st.error("❌ Could not create text chunks")
                return False
            
            # Filter out very short chunks
            valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 20]
            
            if not valid_chunks:
                st.error("❌ No valid text chunks created")
                return False
            
            # Limit chunks for efficiency but keep more for better coverage
            limited_chunks = valid_chunks[:100]  # Increased from 30 to 100
            
            st.info(f"📊 Processing {len(limited_chunks)} text chunks...")
            
            # Create vectorstore with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.vectorstore = FAISS.from_texts(
                        limited_chunks, 
                        self.embeddings
                    )
                    st.success(f"✅ Successfully created knowledge base with {len(limited_chunks)} chunks!")
                    return True
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.warning(f"Attempt {attempt + 1} failed, retrying... ({str(e)})")
                        continue
                    else:
                        st.error(f"❌ Failed to create vectorstore after {max_retries} attempts: {str(e)}")
                        return False
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            return False
    
    def query_financial_data(self, question):
        """Fast financial analysis query"""
        if not self.vectorstore:
            return "Please upload a PDF document first."
        
        try:
            # Get relevant context quickly
            docs = self.vectorstore.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Financial analysis prompt
            prompt = f"""
You are a financial analyst. Analyze the following financial document context and answer the question.

FINANCIAL CONTEXT:
{context}

QUESTION: {question}

Provide a concise financial analysis focusing on:
- Key financial metrics and numbers
- Trends and patterns
- Risks and opportunities
- Actionable insights

Answer:"""

            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Analysis error: {str(e)}"

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = LightweightFinancialRAG()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Sidebar
with st.sidebar:
    st.markdown("### 🔑 Configuration")
    
    api_key = st.text_input(
        "Google API Key",
        type="password",
        help="Get your API key from https://ai.google.dev/"
    )
    
    st.markdown("### 📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload Financial PDF",
        type=['pdf'],
        help="Upload financial reports, statements, or analysis documents"
    )
    
    # Auto-process when file uploaded
    if uploaded_file and api_key and not st.session_state.pdf_processed:
        # Initialize models
        if not st.session_state.rag_system.is_ready:
            with st.spinner("🔄 Initializing..."):
                if st.session_state.rag_system.initialize_models(api_key):
                    st.success("✅ Ready!")
                else:
                    st.stop()
        
        # Process PDF
        with st.spinner("📄 Processing PDF..."):
            if st.session_state.rag_system.process_and_store_fast(uploaded_file):
                st.session_state.pdf_processed = True
                st.success("✅ PDF Ready for Analysis!")
                st.rerun()
            else:
                st.error("❌ Processing failed")
    
    # Reset button
    if st.button("🔄 Reset Chat"):
        st.session_state.chat_history = []
        st.session_state.pdf_processed = False
        st.session_state.rag_system = LightweightFinancialRAG()
        st.rerun()
    
    # Status indicator
    if st.session_state.pdf_processed:
        st.success("🟢 System Ready")
    elif api_key and uploaded_file:
        st.warning("🟡 Processing...")
    else:
        st.info("🔴 Waiting for inputs")

# Main chat interface
st.markdown('<h1 class="main-header">💰 Financial Analyst Assistant</h1>', unsafe_allow_html=True)

# Chat container
chat_container = st.container()

with chat_container:
    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message">🧑‍💼 <strong>You:</strong><br>{message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message assistant-message">💰 <strong>Financial Analyst:</strong><br>{message["content"]}</div>',
                    unsafe_allow_html=True
                )
    else:
        # Welcome message
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <h3>Welcome to Financial Analyst Assistant! 💰</h3>
            <p>Upload a financial PDF and start asking questions about:</p>
            <ul style="list-style: none; padding: 0;">
                <li>📊 Financial metrics and ratios</li>
                <li>📈 Revenue and growth trends</li>
                <li>⚠️ Risk factors and analysis</li>
                <li>💡 Investment insights</li>
                <li>🔍 Comparative analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Chat input at bottom
st.markdown('<div class="chat-input">', unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])

with col1:
    user_question = st.text_input(
        "Ask about the financial document...",
        placeholder="e.g., What are the key financial metrics? What are the main risks?",
        key="user_input",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Send 📤", type="primary")

st.markdown('</div>', unsafe_allow_html=True)

# Process user input
if (send_button or user_question) and user_question:
    if not st.session_state.pdf_processed:
        st.warning("⚠️ Please upload and process a PDF document first!")
    else:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        # Generate response
        with st.spinner("🤔 Analyzing..."):
            response = st.session_state.rag_system.query_financial_data(user_question)
            
            # Add assistant response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
        
        # Clear input and refresh
        st.rerun()

# Quick action buttons for common queries
if st.session_state.pdf_processed:
    st.markdown("### 🚀 Quick Analysis")
    
    quick_questions = [
        "What are the key financial highlights?",
        "What are the main risk factors?",
        "How is the company's financial performance?",
        "What are the revenue trends?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        col_idx = i % 2
        with cols[col_idx]:
            if st.button(question, key=f"quick_{i}"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question
                })
                
                with st.spinner("🤔 Analyzing..."):
                    response = st.session_state.rag_system.query_financial_data(question)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        💰 Financial Analyst Assistant - Powered by Google Gemini & LangChain
    </div>
    """,
    unsafe_allow_html=True
)
