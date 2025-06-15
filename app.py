import streamlit as st
import os
import tempfile
import base64
from typing import List, Dict, Any
import pandas as pd
from PIL import Image
import io

# LangChain imports - Updated to use new packages
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Unstructured imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

import google.generativeai as genai

# Configure page
st.set_page_config(
    page_title="Chat with your database",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Chat UI CSS
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 1rem;}
    .stApp > header {display: none;}
    .stDeployButton {display: none;}
    #MainMenu {display: none;}
    footer {display: none;}
    
    /* Main container */
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Chat title */
    .chat-title {
        text-align: center;
        color: white;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 3rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        max-height: 70vh;
        overflow-y: auto;
        margin-bottom: 2rem;
    }
    
    /* Message bubbles */
    .message-bubble {
        display: flex;
        align-items: flex-start;
        margin: 1.5rem 0;
        gap: 0.75rem;
    }
    
    .message-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: 600;
        flex-shrink: 0;
        margin-top: 4px;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        color: white;
    }
    
    .assistant-avatar {
        background: linear-gradient(135deg, #4ecdc4, #44a08d);
        color: white;
    }
    
    .message-content {
        background: #f8f9fa;
        padding: 1rem 1.25rem;
        border-radius: 18px;
        max-width: 75%;
        line-height: 1.5;
        color: #2c3e50;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .user-message .message-content {
        background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        color: white;
        margin-left: auto;
    }
    
    .assistant-message .message-content {
        background: linear-gradient(135deg, #4ecdc4, #44a08d);
        color: white;
    }
    
    .user-message {
        flex-direction: row-reverse;
    }
    
    /* Input area */
    .input-container {
        background: white;
        border-radius: 50px;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .stTextInput > div > div > input {
        border: none !important;
        outline: none !important;
        background: transparent !important;
        font-size: 16px !important;
        padding: 0.5rem 0 !important;
    }
    
    .stTextInput > div > div {
        border: none !important;
        background: transparent !important;
    }
    
    .stTextInput {
        flex: 1;
    }
    
    /* Send button */
    .send-button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 18px;
        transition: all 0.3s ease;
    }
    
    .send-button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sample questions */
    .sample-questions {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        margin-bottom: 2rem;
    }
    
    .sample-question {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1rem 1.25rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 2px solid transparent;
    }
    
    .sample-question:hover {
        background: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .sample-question-text {
        color: #2c3e50;
        font-weight: 500;
        margin: 0;
    }
    
    /* Upload area */
    .upload-area {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .upload-text {
        color: #2c3e50;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    /* Processing status */
    .processing-status {
        background: rgba(76, 175, 80, 0.1);
        border: 2px solid rgba(76, 175, 80, 0.3);
        border-radius: 15px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #2e7d32;
        font-weight: 500;
        text-align: center;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 3px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 3px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* API Key input styling */
    .api-key-container {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .api-key-text {
        color: #2c3e50;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_embeddings():
    """Initialize embeddings with caching"""
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None

class FinancialRAGProcessor:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.documents = []
        self.extracted_tables = []
        self.extracted_images = []
        self.chat_history = []
        self.is_initialized = False
        
    def initialize_models(self, google_api_key: str):
        """Initialize the LLM and embeddings"""
        if self.is_initialized:
            return True
            
        try:
            # Configure Google AI
            genai.configure(api_key=google_api_key)
            
            # Initialize LLM with multimodal capabilities
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=google_api_key,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            
            # Initialize embeddings
            self.embeddings = initialize_embeddings()
            if not self.embeddings:
                return False
            
            # Initialize chat history
            self.chat_history = []
            
            self.is_initialized = True
            return True
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            return False
    
    def extract_pdf_content(self, pdf_file) -> Dict[str, Any]:
        """Extract comprehensive content from PDF including text, images, and tables - OPTIMIZED"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Create temp directory for images
            image_output_dir = tempfile.mkdtemp()
            
            # OPTIMIZED: Simplified PDF processing for speed
            elements = partition_pdf(
                filename=tmp_file_path,
                extract_images_in_pdf=False,  # Disable image extraction for speed
                infer_table_structure=False,  # Disable table inference for speed
                strategy="fast",  # Use fast strategy
                max_characters=1000,  # Much smaller for faster processing
                new_after_n_chars=900,
                combine_text_under_n_chars=500
            )
            
            # OPTIMIZED: Simplified content separation
            text_elements = []
            table_elements = []
            
            # Just process text elements for speed
            for element in elements:
                content = str(element).strip()
                if content and len(content) > 20:  # Filter out very short content
                    if "table" in content.lower() or "data" in content.lower():
                        table_elements.append(element)
                    else:
                        text_elements.append(element)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return {
                "text_elements": text_elements,
                "table_elements": table_elements,
                "image_elements": [],  # Disabled for speed
                "extracted_image_paths": [],  # Disabled for speed
                "image_output_dir": image_output_dir,
                "total_elements": len(text_elements) + len(table_elements)
            }
            
        except Exception as e:
            st.error(f"Error extracting PDF content: {str(e)}")
            return None
    
    def process_documents(self, extracted_content: Dict[str, Any]):
        """Process extracted content into LangChain documents - OPTIMIZED FOR SPEED"""
        try:
            documents = []
            
            # Process text elements - SIMPLIFIED
            for i, element in enumerate(extracted_content["text_elements"][:50]):  # Limit to first 50 for speed
                content = str(element).strip()
                if content and len(content) > 30:  # Only process meaningful content
                    doc = Document(
                        page_content=content[:800],  # Truncate for speed
                        metadata={
                            "type": "text",
                            "element_id": i,
                            "category": "Financial_Text"
                        }
                    )
                    documents.append(doc)
            
            # Process table elements - SIMPLIFIED
            for i, element in enumerate(extracted_content["table_elements"][:20]):  # Limit to first 20 for speed
                content = str(element).strip()
                if content and len(content) > 30:
                    doc = Document(
                        page_content=f"FINANCIAL DATA: {content[:800]}",  # Truncate for speed
                        metadata={
                            "type": "table",
                            "element_id": i,
                            "category": "Financial_Table"
                        }
                    )
                    documents.append(doc)
                    self.extracted_tables.append({
                        "id": i,
                        "content": content[:500],  # Truncate for display
                        "raw_element": element
                    })
            
            # Skip image processing for speed
            self.extracted_images = []
            
            self.documents = documents
            return len(documents)
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return 0
    
    def create_vectorstore(self):
        """Create FAISS vectorstore from processed documents - SUPER FAST"""
        try:
            if not self.documents:
                st.error("No documents to process")
                return False
            
            # OPTIMIZED: Much smaller chunks for speed
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,  # Very small for speed
                chunk_overlap=50,
                length_function=len
            )
            
            # Limit number of documents for speed
            limited_docs = self.documents[:30]  # Only process first 30 documents
            split_docs = text_splitter.split_documents(limited_docs)
            
            # Create vectorstore with limited documents
            self.vectorstore = FAISS.from_documents(split_docs[:50], self.embeddings)  # Max 50 chunks
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}  # Only top 2 results for speed
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error creating vectorstore: {str(e)}")
            return False
    
    def query_documents(self, question: str) -> str:
        """Query documents - OPTIMIZED VERSION"""
        try:
            if not self.retriever:
                return "Please upload and process a document first."
            
            # Get relevant documents - simplified
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            # Simple context combination
            context_parts = []
            for doc in relevant_docs:
                context_parts.append(doc.page_content[:400])  # Truncate for speed
            
            full_context = "\n\n".join(context_parts)
            
            # Simple prompt without complex formatting
            simple_prompt = f"""
            You are a financial analyst. Answer the question based on the context provided.
            
            Context: {full_context}
            
            Question: {question}
            
            Provide a clear, concise financial analysis answer.
            """
            
            # Generate response
            response = self.llm.invoke(simple_prompt)
            
            # Save to chat history
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": response.content})
            
            return response.content
            
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return "Sorry, I encountered an error while processing your question."

# Initialize session state
if "rag_processor" not in st.session_state:
    st.session_state.rag_processor = FinancialRAGProcessor()

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

if "processing_status" not in st.session_state:
    st.session_state.processing_status = None

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Title
st.markdown('<h1 class="chat-title">Chat with your database</h1>', unsafe_allow_html=True)

# API Key input
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""

if not st.session_state.google_api_key:
    st.markdown("""
    <div class="api-key-container">
        <div class="api-key-text">🔑 Please enter your Google API Key to get started</div>
    </div>
    """, unsafe_allow_html=True)
    
    api_key = st.text_input("", type="password", placeholder="Enter your Google API Key...", key="api_input")
    if api_key:
        st.session_state.google_api_key = api_key
        st.rerun()

# File upload
elif not st.session_state.documents_processed:
    st.markdown("""
    <div class="upload-area">
        <div class="upload-text">📄 Upload your financial document to start analyzing</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=['pdf'], label_visibility="collapsed")
    
    if uploaded_file:
        # Initialize models automatically
        if not st.session_state.rag_processor.is_initialized:
            with st.spinner("🔄 Initializing AI models..."):
                success = st.session_state.rag_processor.initialize_models(st.session_state.google_api_key)
                if success:
                    st.success("✅ Models initialized!")
                else:
                    st.error("❌ Failed to initialize models")
                    st.stop()
        
        # Process document automatically
        if st.session_state.rag_processor.is_initialized:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Extract content
                status_text.text("📄 Extracting content from PDF...")
                progress_bar.progress(20)
                extracted_content = st.session_state.rag_processor.extract_pdf_content(uploaded_file)
                
                if extracted_content:
                    status_text.text(f"✅ Extracted {extracted_content['total_elements']} elements")
                    progress_bar.progress(50)
                    
                    # Step 2: Process documents
                    status_text.text("🔄 Processing documents...")
                    doc_count = st.session_state.rag_processor.process_documents(extracted_content)
                    progress_bar.progress(75)
                    
                    # Step 3: Create vectorstore
                    status_text.text("🧠 Creating knowledge base...")
                    if st.session_state.rag_processor.create_vectorstore():
                        progress_bar.progress(100)
                        status_text.text("✅ Ready for analysis!")
                        st.session_state.documents_processed = True
                        st.session_state.processing_status = f"Processed {doc_count} document chunks successfully!"
                        st.rerun()
                    else:
                        st.error("❌ Failed to create knowledge base")
                else:
                    st.error("❌ Failed to extract content from PDF")
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")

# Chat interface
else:
    # Show processing status
    if st.session_state.processing_status:
        st.markdown(f'<div class="processing-status">🎉 {st.session_state.processing_status}</div>', unsafe_allow_html=True)
    
    # Sample questions (only show if no chat history)
    if not st.session_state.rag_processor.chat_history:
        sample_questions = [
            "What are the company's key financial metrics and ratios?",
            "Analyze the revenue trends and growth patterns over recent periods",
            "What are the main risk factors mentioned in this financial report?",
            "Compare the company's profitability margins with industry standards",
            "What is the company's debt-to-equity ratio and financial leverage position?",
            "Summarize the cash flow statement and liquidity position"
        ]
        
        st.markdown('<div class="sample-questions">', unsafe_allow_html=True)
        for i, question in enumerate(sample_questions):
            if st.button(question, key=f"sample_{i}"):
                st.session_state.current_question = question
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.rag_processor.chat_history:
        if message["role"] == "user":
            st.markdown(f'''
            <div class="message-bubble user-message">
                <div class="message-avatar user-avatar">U</div>
                <div class="message-content">{message["content"]}</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="message-bubble assistant-message">
                <div class="message-avatar assistant-avatar">AI</div>
                <div class="message-content">{message["content"]}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([10, 1])
    
    with col1:
        query = st.text_input(
            "",
            value=st.session_state.get("current_question", ""),
            placeholder="Ask about your data...",
            key="query_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_clicked = st.button("→", key="send_btn", help="Send message")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process query
    if (send_clicked or st.session_state.get("current_question")) and query:
        with st.spinner("🔍 Analyzing..."):
            response = st.session_state.rag_processor.query_documents(query)
            
            # Clear current question
            if "current_question" in st.session_state:
                del st.session_state.current_question
            
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
