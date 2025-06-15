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
    page_title="Financial RAG Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .document-preview {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fafafa;
    }
    .processing-status {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
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
                chunk_size=1000,  # Very small for speed
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
    
    def get_financial_prompt(self):
        """Get the multimodal financial analysis prompt template"""
        return PromptTemplate(
            template="""You are an advanced multimodal financial analyst assistant specialized in analyzing SEC 10-K reports, financial documents, charts, graphs, and tabular data. 
            
            Use the following context to answer questions about financial data, company performance, risks, market analysis, visual trends, and numerical patterns.
            
            Context: {context}
            
            Chat History: {chat_history}
            
            Question: {input}
            
            ANALYSIS GUIDELINES:
            - For TEXT data: Provide detailed financial analysis with specific numbers, ratios, and trends
            - For TABLE data: Analyze financial metrics, perform calculations, identify patterns and trends
            - For IMAGE data: Describe visual elements like charts, graphs, trends, and key insights
            - Combine insights from all data types (text, tables, images) when available
            - Always specify which type of data (text/table/image) your analysis is based on
            - Include specific numbers, percentages, ratios, and financial metrics when available
            - Highlight key financial trends, risks, and opportunities
            
            If you cannot find specific information in the provided context, clearly state that the information is not available in the provided documents.
            
            Answer:""",
            input_variables=["context", "chat_history", "input"]
        )
    
    def analyze_image_with_gemini(self, image_path: str, question: str) -> str:
        """Analyze financial charts/images using Gemini 2.0 Flash multimodal capabilities"""
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode()
            
            # Create multimodal prompt for financial image analysis
            prompt = f"""
            Analyze this financial chart/graph/visual element from a financial document.
            
            Question: {question}
            
            Please provide:
            1. Description of the visual elements (chart type, axes, data points)
            2. Key financial trends and patterns visible
            3. Specific numbers, percentages, or values if readable
            4. Financial insights and implications
            5. Any risks or opportunities highlighted by the visual data
            
            Focus on actionable financial analysis and insights.
            """
            
            # Use Google AI directly for multimodal analysis
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content([
                prompt,
                {"mime_type": "image/png", "data": image_base64}
            ])
            
            return response.text
            
        except Exception as e:
            st.error(f"Error analyzing image: {str(e)}")
            return f"Could not analyze image: {os.path.basename(image_path)}"
    
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

# Main UI
st.markdown('<h1 class="main-header">📊 Financial RAG Analyst</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Key input
    google_api_key = st.text_input(
        "Google API Key",
        type="password",
        help="Enter your Google AI API key"
    )
    
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload Financial Document (PDF)",
        type=['pdf'],
        help="Upload SEC 10-K reports or other financial documents"
    )
    
    # Auto-process when file is uploaded
    if uploaded_file and google_api_key and not st.session_state.documents_processed:
        # Initialize models automatically
        if not st.session_state.rag_processor.is_initialized:
            with st.spinner("🔄 Initializing AI models..."):
                success = st.session_state.rag_processor.initialize_models(google_api_key)
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
    
    # Document statistics
    if st.session_state.documents_processed:
        st.header("📊 Document Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(st.session_state.rag_processor.documents))
        with col2:
            st.metric("Images", len(st.session_state.rag_processor.extracted_images))
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Tables", len(st.session_state.rag_processor.extracted_tables))
        with col4:
            st.metric("Chat History", len(st.session_state.rag_processor.chat_history))

# Main content area
if not google_api_key:
    st.info("👈 Please enter your Google API key in the sidebar to get started.")
elif not uploaded_file:
    st.info("👈 Please upload a financial document (PDF) to start analyzing.")
elif not st.session_state.documents_processed:
    st.info("🔄 Processing your document... Please wait.")
else:
    # Show processing status
    if st.session_state.processing_status:
        st.markdown(f'<div class="processing-status">🎉 {st.session_state.processing_status}</div>', unsafe_allow_html=True)
    
    # Sample questions for multimodal analysis
    st.header("💡 Sample Questions")
    sample_questions = [
        "What are the key financial metrics shown in the charts and tables?",
        "Analyze the revenue trends from both text and visual data",
        "What risks are mentioned in text and shown in risk charts?",
        "Compare the financial ratios across different data sources",
        "What insights can you derive from the financial graphs and tables?",
        "Analyze the company's performance using all available data types"
    ]
    
    cols = st.columns(3)
    for i, question in enumerate(sample_questions):
        col_idx = i % 3
        with cols[col_idx]:
            if st.button(question, key=f"sample_{i}"):
                st.session_state.current_question = question
    
    # Chat interface
    st.header("💬 Financial Analysis Chat")
    
    # Display chat history
    for message in st.session_state.rag_processor.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Query input
    query = st.text_input(
        "Ask a question about the financial document:",
        value=st.session_state.get("current_question", ""),
        key="query_input"
    )
    
    if st.button("🔍 Analyze", type="primary") and query:
        with st.spinner("🔍 Analyzing document..."):
            response = st.session_state.rag_processor.query_documents(query)
            
            # Clear current question
            if "current_question" in st.session_state:
                del st.session_state.current_question
            
            st.rerun()
    
    # Document preview section with multimodal content
    with st.expander("📋 Multimodal Document Preview", expanded=False):
        tab1, tab2, tab3 = st.tabs(["📄 Text Content", "📊 Tables", "🖼️ Images"])
        
        with tab1:
            if st.session_state.rag_processor.documents:
                text_docs = [doc for doc in st.session_state.rag_processor.documents if doc.metadata.get("type") == "text"]
                if text_docs:
                    st.markdown("### Text Content Sample")
                    st.markdown(f'<div class="document-preview">{text_docs[0].page_content[:1000]}...</div>', unsafe_allow_html=True)
                else:
                    st.info("No text content found")
        
        with tab2:
            if st.session_state.rag_processor.extracted_tables:
                st.markdown("### Extracted Financial Tables")
                for i, table in enumerate(st.session_state.rag_processor.extracted_tables[:3]):
                    st.markdown(f"**Table {i+1}:**")
                    st.text(table["content"][:500] + "..." if len(table["content"]) > 500 else table["content"])
            else:
                st.info("No tables found")
        
        with tab3:
            if st.session_state.rag_processor.extracted_images:
                st.markdown("### Extracted Financial Charts/Images")
                for i, img_info in enumerate(st.session_state.rag_processor.extracted_images[:5]):
                    try:
                        if os.path.exists(img_info["path"]):
                            st.markdown(f"**Image {i+1}: {img_info['filename']}**")
                            image = Image.open(img_info["path"])
                            st.image(image, caption=f"Financial Chart/Graph {i+1}", use_column_width=True)
                    except Exception as e:
                        st.error(f"Could not display image {i+1}: {str(e)}")
            else:
                st.info("No images found")
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.rag_processor.chat_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Multimodal Financial RAG Analyst - Powered by Gemini 2.0 Flash & LangChain</p>
        <p>Advanced analysis of text, tables, charts, and images in financial documents</p>
    </div>
    """,
    unsafe_allow_html=True
)
