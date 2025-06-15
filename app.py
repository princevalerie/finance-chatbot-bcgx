import streamlit as st
from PyPDF2 import PdfReader
import base64
import os
import tempfile
from datetime import datetime
import traceback

# LangChain imports - minimal and efficient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
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
    .error-details {
        background-color: #ffe6e6;
        border: 1px solid #ff9999;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        font-family: monospace;
        font-size: 0.9em;
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
        """Initialize models with better error handling"""
        if self.is_ready:
            return True
            
        try:
            # Test the API key first
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.3
            )
            
            # Quick test
            test_response = self.llm.invoke("Hello")
            
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
            
            self.is_ready = True
            st.success("✅ Models initialized successfully!")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
                st.error("❌ Invalid Google API Key. Please check your API key from https://ai.google.dev/")
            elif "quota" in error_msg.lower():
                st.error("❌ API quota exceeded. Check your usage limits.")
            else:
                st.error(f"❌ Model initialization failed: {error_msg}")
            
            with st.expander("🔍 See error details"):
                st.code(traceback.format_exc())
            return False
    
    def extract_pdf_text_fast(self, pdf_file):
        """Enhanced PDF text extraction with better error handling"""
        try:
            # Reset file pointer
            pdf_file.seek(0)
            
            # Try PyPDF2 first
            try:
                st.info("📄 Extracting text from PDF...")
                pdf_reader = PdfReader(pdf_file)
                
                # Check if encrypted
                if pdf_reader.is_encrypted:
                    st.warning("⚠️ PDF is encrypted, trying to decrypt...")
                    try:
                        pdf_reader.decrypt("")
                    except:
                        st.error("❌ Cannot decrypt PDF")
                        return ""
                
                text = ""
                total_pages = len(pdf_reader.pages)
                st.info(f"📄 Processing {total_pages} pages...")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text + "\n"
                    except Exception as page_error:
                        st.warning(f"⚠️ Could not read page {page_num + 1}")
                        continue
                
                if text.strip():
                    st.success(f"✅ Extracted {len(text)} characters from PDF")
                    return text
                else:
                    st.error("❌ No readable text found in PDF")
                    return ""
                    
            except Exception as e:
                st.error(f"❌ PDF reading failed: {str(e)}")
                return ""
                
        except Exception as e:
            st.error(f"❌ PDF extraction failed: {str(e)}")
            with st.expander("🔍 See error details"):
                st.code(traceback.format_exc())
            return ""
    
    def process_and_store_fast(self, pdf_file):
        """Enhanced processing with better error handling"""
        try:
            # Extract text
            raw_text = self.extract_pdf_text_fast(pdf_file)
            
            if not raw_text or len(raw_text.strip()) < 50:
                st.error("❌ Insufficient text content in PDF")
                return False
            
            # Create chunks
            st.info("🔄 Creating text chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=150,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            chunks = text_splitter.split_text(raw_text)
            
            if not chunks:
                st.error("❌ Could not create text chunks")
                return False
            
            # Filter valid chunks
            valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 30]
            
            if not valid_chunks:
                st.error("❌ No valid chunks created")
                return False
            
            # Limit for efficiency
            final_chunks = valid_chunks[:80]
            st.info(f"📊 Processing {len(final_chunks)} chunks...")
            
            # Create vectorstore with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    progress = st.progress(0)
                    progress.progress(50)
                    
                    self.vectorstore = FAISS.from_texts(
                        final_chunks, 
                        self.embeddings
                    )
                    
                    progress.progress(100)
                    st.success(f"✅ Knowledge base created with {len(final_chunks)} chunks!")
                    
                    # Test the vectorstore
                    test_results = self.vectorstore.similarity_search("test", k=1)
                    if test_results:
                        st.info("✅ System ready for questions!")
                    
                    return True
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.warning(f"⚠️ Attempt {attempt + 1} failed, retrying...")
                        continue
                    else:
                        st.error(f"❌ Vectorstore creation failed: {str(e)}")
                        
                        if "quota" in str(e).lower():
                            st.error("🚨 API quota exceeded - please check your Google AI usage")
                        elif "api" in str(e).lower():
                            st.error("🚨 API error - please verify your API key")
                        
                        with st.expander("🔍 See error details"):
                            st.code(traceback.format_exc())
                        return False
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            with st.expander("🔍 See error details"):
                st.code(traceback.format_exc())
            return False
    
    def query_financial_data(self, question):
        """Enhanced financial query with better error handling"""
        if not self.vectorstore:
            return "❌ Please upload and process a PDF document first."
        
        try:
            # Get relevant context
            docs = self.vectorstore.similarity_search(question, k=4)
            
            if not docs:
                return "❌ No relevant information found for your question."
            
            context = "\n".join([doc.page_content for doc in docs])
            
            # Enhanced prompt
            prompt = f"""
You are a professional financial analyst. Based on the document context provided, answer the user's question with detailed financial analysis.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

Please provide a comprehensive analysis including:
- Direct answer to the question
- Key financial metrics and numbers
- Trends and patterns
- Business implications
- Actionable insights

Keep your response clear, structured, and professional.

ANALYSIS:"""

            try:
                response = self.llm.invoke(prompt)
                return response.content
            except Exception as llm_error:
                return f"❌ Analysis error: {str(llm_error)}. Please try rephrasing your question."
            
        except Exception as e:
            return f"❌ Query error: {str(e)}"

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = LightweightFinancialRAG()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "processing" not in st.session_state:
    st.session_state.processing = False

# Sidebar
with st.sidebar:
    st.markdown("### 🔑 Configuration")
    
    api_key = st.text_input(
        "Google API Key",
        type="password",
        help="Get your API key from https://ai.google.dev/",
        placeholder="Enter your Google API key..."
    )
    
    if api_key:
        st.success("✅ API Key provided")
    
    st.markdown("### 📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload Financial PDF",
        type=['pdf'],
        help="Upload financial reports, statements, or documents"
    )
    
    # Manual processing button
    if uploaded_file and api_key and not st.session_state.pdf_processed and not st.session_state.processing:
        if st.button("🚀 Process PDF", type="primary"):
            st.session_state.processing = True
            st.rerun()
    
    # Processing logic
    if st.session_state.processing and uploaded_file and api_key:
        try:
            # Initialize models
            if not st.session_state.rag_system.is_ready:
                with st.spinner("🔄 Initializing AI models..."):
                    if not st.session_state.rag_system.initialize_models(api_key):
                        st.session_state.processing = False
                        st.stop()
            
            # Process PDF
            with st.spinner("📄 Processing PDF..."):
                if st.session_state.rag_system.process_and_store_fast(uploaded_file):
                    st.session_state.pdf_processed = True
                    st.session_state.processing = False
                    st.success("🎉 PDF processed successfully!")
                    st.rerun()
                else:
                    st.session_state.processing = False
                    st.error("❌ PDF processing failed")
                    
        except Exception as e:
            st.session_state.processing = False
            st.error(f"❌ Unexpected error: {str(e)}")
            with st.expander("🔍 Error details"):
                st.code(traceback.format_exc())
    
    # Reset button
    if st.button("🔄 Reset All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Status
    st.markdown("### 📊 Status")
    if st.session_state.processing:
        st.warning("🟡 Processing...")
    elif st.session_state.pdf_processed:
        st.success("🟢 Ready for Analysis")
    elif api_key and uploaded_file:
        st.info("🔵 Ready to Process")
    elif api_key:
        st.info("📄 Upload PDF")
    else:
        st.warning("🔑 Enter API Key")

# Main interface
st.markdown('<h1 class="main-header">💰 Financial Analyst Assistant</h1>', unsafe_allow_html=True)

# Chat container
if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f'<div class="chat-message user-message">🧑‍💼 <strong>You:</strong><br>{message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-message assistant-message">💰 <strong>Analyst:</strong><br>{message["content"]}</div>',
                unsafe_allow_html=True
            )
else:
    # Welcome message
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <h3>Welcome to Financial Analyst Assistant! 💰</h3>
        <p>Upload a financial PDF and get professional analysis on:</p>
        <ul style="list-style: none; padding: 0;">
            <li>📊 Financial metrics and KPIs</li>
            <li>📈 Revenue and growth analysis</li>
            <li>⚠️ Risk assessment</li>
            <li>💡 Investment insights</li>
            <li>🔍 Performance evaluation</li>
        </ul>
        <p><strong>Steps to get started:</strong></p>
        <ol style="text-align: left; max-width: 400px; margin: 0 auto;">
            <li>Enter your Google API key</li>
            <li>Upload a financial PDF</li>
            <li>Click "Process PDF"</li>
            <li>Start asking questions!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Chat input
if not st.session_state.processing:
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask about the financial document...",
            placeholder="What are the key financial metrics? How is performance?",
            key="user_input",
            label_visibility="collapsed",
            disabled=not st.session_state.pdf_processed
        )
    
    with col2:
        send_button = st.button("Send 📤", type="primary", disabled=not st.session_state.pdf_processed)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle user input
    if (send_button or user_question) and user_question and st.session_state.pdf_processed:
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
        
        st.rerun()

# Quick questions
if st.session_state.pdf_processed and not st.session_state.processing:
    st.markdown("### 🚀 Quick Analysis")
    
    questions = [
        "What are the key financial highlights?",
        "What are the main risks?",
        "How is the financial performance?",
        "What are revenue trends?",
        "What are the major expenses?",
        "Any investment insights?"
    ]
    
    cols = st.columns(3)
    for i, question in enumerate(questions):
        with cols[i % 3]:
            if st.button(question, key=f"q_{i}"):
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
        💰 Financial Analyst Assistant - Enhanced Error Handling<br>
        Powered by Google Gemini & LangChain
    </div>
    """,
    unsafe_allow_html=True
)
