import streamlit as st
from PyPDF2 import PdfReader
import base64
import os
import tempfile
from datetime import datetime
import traceback
import logging

# LangChain imports - minimal and efficient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.processed_text = None
        
    def initialize_models(self, api_key):
        """Initialize models with minimal setup"""
        if self.is_ready:
            return True
            
        try:
            # Test API key validity first
            test_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.3
            )
            
            # Simple test query
            test_response = test_llm.invoke("Hello")
            
            self.llm = test_llm
            
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
            
            # Test embeddings
            test_embedding = self.embeddings.embed_query("test")
            
            self.is_ready = True
            st.success("✅ Models initialized successfully!")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "invalid API key" in error_msg.lower():
                st.error("❌ Invalid Google API Key. Please check your API key.")
            elif "quota" in error_msg.lower():
                st.error("❌ API quota exceeded. Please check your Google AI Studio usage.")
            else:
                st.error(f"❌ Model initialization failed: {error_msg}")
            
            # Show detailed error in expander
            with st.expander("🔍 See detailed error"):
                st.markdown(f'<div class="error-details">{traceback.format_exc()}</div>', 
                          unsafe_allow_html=True)
            return False
    
    def extract_pdf_text_robust(self, pdf_file):
        """Robust PDF text extraction with multiple fallback methods"""
        text_content = ""
        
        try:
            # Reset file pointer
            pdf_file.seek(0)
            
            # Method 1: PyPDF2 (most reliable for text)
            st.info("📄 Attempting text extraction with PyPDF2...")
            try:
                pdf_reader = PdfReader(pdf_file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    st.warning("⚠️ PDF is encrypted, attempting to decrypt...")
                    try:
                        pdf_reader.decrypt("")  # Try empty password
                    except:
                        st.error("❌ Cannot decrypt PDF. Please provide an unencrypted version.")
                        return ""
                
                total_pages = len(pdf_reader.pages)
                st.info(f"📄 Found {total_pages} pages in PDF")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_content += f"\n=== Page {page_num + 1} ===\n"
                            text_content += page_text.strip() + "\n"
                            
                        # Update progress
                        if page_num % 5 == 0:
                            st.info(f"📄 Processed {page_num + 1}/{total_pages} pages...")
                            
                    except Exception as page_error:
                        st.warning(f"⚠️ Could not read page {page_num + 1}: {str(page_error)}")
                        continue
                
                if text_content.strip():
                    st.success(f"✅ Successfully extracted {len(text_content)} characters using PyPDF2")
                    return text_content
                else:
                    st.warning("⚠️ PyPDF2 extracted no readable text")
                    
            except Exception as e:
                st.warning(f"⚠️ PyPDF2 failed: {str(e)}")
            
            # Method 2: Try with pdfplumber if available
            try:
                import pdfplumber
                st.info("📄 Trying pdfplumber for better table extraction...")
                
                pdf_file.seek(0)
                with pdfplumber.open(pdf_file) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text_content += f"\n=== Page {page_num + 1} ===\n"
                                text_content += page_text + "\n"
                        except Exception as page_error:
                            continue
                
                if text_content.strip():
                    st.success(f"✅ Successfully extracted {len(text_content)} characters using pdfplumber")
                    return text_content
                    
            except ImportError:
                st.info("💡 Install pdfplumber for better PDF processing: pip install pdfplumber")
            except Exception as e:
                st.warning(f"⚠️ pdfplumber failed: {str(e)}")
            
            # Method 3: Alternative PyPDF2 approach
            try:
                st.info("📄 Trying alternative extraction method...")
                pdf_file.seek(0)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Try reading with different parameters
                with open(tmp_file_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        try:
                            # Try different extraction methods
                            text = page.extract_text(extraction_mode="layout")
                            if not text:
                                text = page.extract_text()
                            if text:
                                text_content += text + "\n"
                        except:
                            continue
                
                # Clean up
                os.unlink(tmp_file_path)
                
                if text_content.strip():
                    st.success(f"✅ Alternative method extracted {len(text_content)} characters")
                    return text_content
                    
            except Exception as e:
                st.warning(f"⚠️ Alternative method failed: {str(e)}")
            
            # If all methods fail
            if not text_content.strip():
                st.error("❌ Could not extract any readable text from PDF")
                st.info("💡 This might be a scanned PDF. Consider using OCR tools or converting to text first.")
                return ""
                
            return text_content
            
        except Exception as e:
            st.error(f"❌ PDF extraction completely failed: {str(e)}")
            with st.expander("🔍 See detailed error"):
                st.markdown(f'<div class="error-details">{traceback.format_exc()}</div>', 
                          unsafe_allow_html=True)
            return ""
    
    def process_and_store_robust(self, pdf_file):
        """Enhanced processing with comprehensive error handling"""
        try:
            # Extract text with detailed progress
            st.info("🔄 Starting PDF processing...")
            raw_text = self.extract_pdf_text_robust(pdf_file)
            
            if not raw_text or len(raw_text.strip()) < 50:
                st.error("❌ Insufficient text content extracted from PDF")
                return False
            
            # Store the processed text for debugging
            self.processed_text = raw_text
            st.success(f"✅ Text extraction complete: {len(raw_text)} characters")
            
            # Create chunks with progress indication
            st.info("🔄 Creating text chunks for analysis...")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Increased chunk size for better context
                chunk_overlap=200,  # More overlap for continuity
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
            )
            
            # Split text into chunks
            chunks = text_splitter.split_text(raw_text)
            
            if not chunks:
                st.error("❌ Could not create text chunks")
                return False
            
            # Filter and validate chunks
            valid_chunks = []
            for chunk in chunks:
                cleaned_chunk = chunk.strip()
                if len(cleaned_chunk) > 50:  # Only keep substantial chunks
                    valid_chunks.append(cleaned_chunk)
            
            if not valid_chunks:
                st.error("❌ No valid text chunks created")
                return False
            
            # Limit chunks for efficiency while keeping good coverage
            max_chunks = min(150, len(valid_chunks))  # Increased limit
            final_chunks = valid_chunks[:max_chunks]
            
            st.info(f"📊 Processing {len(final_chunks)} text chunks...")
            
            # Create vectorstore with robust error handling
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Create FAISS vectorstore
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Creating embeddings...")
                    progress_bar.progress(30)
                    
                    self.vectorstore = FAISS.from_texts(
                        final_chunks, 
                        self.embeddings,
                        metadatas=[{"chunk_id": i, "source": "pdf"} for i in range(len(final_chunks))]
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Vectorstore created successfully!")
                    
                    st.success(f"✅ Knowledge base ready with {len(final_chunks)} chunks!")
                    
                    # Test the vectorstore
                    test_results = self.vectorstore.similarity_search("financial", k=1)
                    if test_results:
                        st.info("✅ Vectorstore test successful - ready for queries!")
                    
                    return True
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    if attempt < max_retries - 1:
                        st.warning(f"⚠️ Attempt {attempt + 1} failed, retrying... ({error_msg})")
                        continue
                    else:
                        st.error(f"❌ Failed to create vectorstore after {max_retries} attempts")
                        
                        # Provide specific error guidance
                        if "api" in error_msg.lower() or "quota" in error_msg.lower():
                            st.error("🚨 API Error: Check your Google API key and quota limits")
                        elif "memory" in error_msg.lower():
                            st.error("🚨 Memory Error: Try with a smaller PDF or restart the app")
                        else:
                            st.error(f"🚨 Vectorstore Error: {error_msg}")
                        
                        with st.expander("🔍 See detailed error"):
                            st.markdown(f'<div class="error-details">{traceback.format_exc()}</div>', 
                                      unsafe_allow_html=True)
                        return False
            
        except Exception as e:
            st.error(f"❌ Processing failed with unexpected error: {str(e)}")
            with st.expander("🔍 See detailed error"):
                st.markdown(f'<div class="error-details">{traceback.format_exc()}</div>', 
                          unsafe_allow_html=True)
            return False
    
    def query_financial_data(self, question):
        """Enhanced financial analysis query with better context handling"""
        if not self.vectorstore:
            return "❌ Please upload and process a PDF document first."
        
        try:
            # Get relevant context with more results for better coverage
            relevant_docs = self.vectorstore.similarity_search_with_score(question, k=5)
            
            if not relevant_docs:
                return "❌ No relevant information found in the document."
            
            # Filter by relevance score and combine context
            context_parts = []
            for doc, score in relevant_docs:
                if score < 0.8:  # Only include highly relevant chunks
                    context_parts.append(doc.page_content)
            
            if not context_parts:
                # Fallback to top results even if scores are high
                context_parts = [doc.page_content for doc, _ in relevant_docs[:3]]
            
            context = "\n\n".join(context_parts)
            
            # Enhanced financial analysis prompt
            prompt = f"""
You are an expert financial analyst. Analyze the provided document context to answer the user's question comprehensively.

DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

Please provide a detailed financial analysis that includes:

1. **Direct Answer**: Address the specific question asked
2. **Key Metrics**: Highlight relevant financial numbers, ratios, and indicators
3. **Analysis**: Interpret what these metrics mean for financial health
4. **Trends**: Identify any patterns or changes over time
5. **Context**: Explain the broader implications
6. **Recommendations**: Suggest actionable insights where appropriate

Format your response clearly with headers and bullet points where helpful.

ANALYSIS:"""

            # Generate response with error handling
            try:
                response = self.llm.invoke(prompt)
                return response.content
            except Exception as llm_error:
                return f"❌ Analysis error: {str(llm_error)}. Please try rephrasing your question."
            
        except Exception as e:
            return f"❌ Query processing error: {str(e)}"

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = LightweightFinancialRAG()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "processing_in_progress" not in st.session_state:
    st.session_state.processing_in_progress = False

# Sidebar configuration
with st.sidebar:
    st.markdown("### 🔑 Configuration")
    
    api_key = st.text_input(
        "Google API Key",
        type="password",
        help="Get your API key from https://ai.google.dev/",
        placeholder="Enter your Google API key here..."
    )
    
    if api_key:
        st.success("✅ API Key provided")
    else:
        st.info("🔑 Please enter your Google API key")
    
    st.markdown("### 📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload Financial PDF",
        type=['pdf'],
        help="Upload financial reports, statements, or analysis documents (max 200MB)"
    )
    
    # Manual process button for better control
    if uploaded_file and api_key and not st.session_state.pdf_processed and not st.session_state.processing_in_progress:
        if st.button("🚀 Process PDF", type="primary"):
            st.session_state.processing_in_progress = True
            st.rerun()
    
    # Processing logic
    if st.session_state.processing_in_progress and uploaded_file and api_key:
        # Initialize models first
        if not st.session_state.rag_system.is_ready:
            with st.spinner("🔄 Initializing AI models..."):
                if st.session_state.rag_system.initialize_models(api_key):
                    st.success("✅ AI models ready!")
                else:
                    st.session_state.processing_in_progress = False
                    st.stop()
        
        # Process PDF
        with st.spinner("📄 Processing PDF document..."):
            if st.session_state.rag_system.process_and_store_robust(uploaded_file):
                st.session_state.pdf_processed = True
                st.session_state.processing_in_progress = False
                st.success("🎉 PDF processing complete! You can now ask questions.")
                st.rerun()
            else:
                st.session_state.processing_in_progress = False
                st.error("❌ PDF processing failed. Please check the errors above.")
    
    # Reset button
    if st.button("🔄 Reset Everything"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Debug information
    if st.session_state.pdf_processed and st.checkbox("🔍 Show Debug Info"):
        st.markdown("### 📊 Debug Information")
        if st.session_state.rag_system.processed_text:
            text_length = len(st.session_state.rag_system.processed_text)
            st.info(f"Extracted text: {text_length:,} characters")
            
            # Show first 500 characters
            with st.expander("Preview extracted text"):
                st.text(st.session_state.rag_system.processed_text[:500] + "...")
    
    # Status indicator
    st.markdown("### 📊 Status")
    if st.session_state.processing_in_progress:
        st.warning("🟡 Processing in progress...")
    elif st.session_state.pdf_processed:
        st.success("🟢 System Ready for Analysis")
    elif api_key and uploaded_file:
        st.info("🔵 Ready to process - click 'Process PDF' button")
    elif api_key:
        st.info("📄 Please upload a PDF document")
    else:
        st.warning("🔑 Please enter your API key")

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
                <li>💼 Business performance evaluation</li>
            </ul>
            <p><strong>To get started:</strong></p>
            <ol style="list-style: none; padding: 0;">
                <li>1️⃣ Enter your Google API key in the sidebar</li>
                <li>2️⃣ Upload a financial PDF document</li>
                <li>3️⃣ Click 'Process PDF' to analyze the document</li>
                <li>4️⃣ Start asking questions!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# Chat input at bottom
if not st.session_state.processing_in_progress:
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask about the financial document...",
            placeholder="e.g., What are the key financial metrics? What are the main risks? How is the revenue trend?",
            key="user_input",
            label_visibility="collapsed",
            disabled=not st.session_state.pdf_processed
        )
    
    with col2:
        send_button = st.button(
            "Send 📤", 
            type="primary",
            disabled=not st.session_state.pdf_processed
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process user input
    if (send_button or user_question) and user_question:
        if not st.session_state.pdf_processed:
            st.warning("⚠️ Please process a PDF document first!")
        else:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            
            # Generate response
            with st.spinner("🤔 Analyzing your question..."):
                response = st.session_state.rag_system.query_financial_data(user_question)
                
                # Add assistant response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
            
            # Clear input and refresh
            st.rerun()

# Quick action buttons for common queries
if st.session_state.pdf_processed and not st.session_state.processing_in_progress:
    st.markdown("### 🚀 Quick Analysis Questions")
    
    quick_questions = [
        "What are the key financial highlights and metrics?",
        "What are the main risk factors mentioned?",
        "How is the company's financial performance trending?",
        "What are the revenue and profit trends?",
        "What are the major expenses and cost drivers?",
        "What insights can you provide about cash flow?"
    ]
    
    cols = st.columns(3)
    for i, question in enumerate(quick_questions):
        col_idx = i % 3
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
        💰 Financial Analyst Assistant - Enhanced with robust error handling<br>
        Powered by Google Gemini & LangChain | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
