import streamlit as st
import os
import tempfile
import base64
from typing import List, Dict, Any
import pandas as pd
from PIL import Image
import io

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

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
</style>
""", unsafe_allow_html=True)

class FinancialRAGProcessor:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.memory = None
        self.documents = []
        self.extracted_tables = []
        self.extracted_images = []
        
    def initialize_models(self, google_api_key: str):
        """Initialize the LLM and embeddings"""
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
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            return True
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            return False
    
    def extract_pdf_content(self, pdf_file) -> Dict[str, Any]:
        """Extract comprehensive content from PDF including text, images, and tables"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Create temp directory for images
            image_output_dir = tempfile.mkdtemp()
            
            # Partition PDF with advanced settings for multimodal extraction
            elements = partition_pdf(
                filename=tmp_file_path,
                extract_images_in_pdf=True,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=4000,
                new_after_n_chars=3800,
                combine_text_under_n_chars=2000,
                image_output_dir_path=image_output_dir
            )
            
            # Separate different types of content
            text_elements = []
            table_elements = []
            image_elements = []
            extracted_image_paths = []
            
            # Get extracted image files
            if os.path.exists(image_output_dir):
                for filename in os.listdir(image_output_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        extracted_image_paths.append(os.path.join(image_output_dir, filename))
            
            for element in elements:
                if hasattr(element, 'category'):
                    if element.category == "Table":
                        table_elements.append(element)
                    elif element.category == "Image":
                        image_elements.append(element)
                    else:
                        text_elements.append(element)
                else:
                    text_elements.append(element)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return {
                "text_elements": text_elements,
                "table_elements": table_elements,
                "image_elements": image_elements,
                "extracted_image_paths": extracted_image_paths,
                "image_output_dir": image_output_dir,
                "total_elements": len(elements)
            }
            
        except Exception as e:
            st.error(f"Error extracting PDF content: {str(e)}")
            return None
    
    def process_documents(self, extracted_content: Dict[str, Any]):
        """Process extracted content into LangChain documents with multimodal support"""
        try:
            documents = []
            
            # Process text elements
            for i, element in enumerate(extracted_content["text_elements"]):
                content = str(element)
                if content.strip():
                    doc = Document(
                        page_content=content,
                        metadata={
                            "type": "text",
                            "element_id": i,
                            "category": getattr(element, 'category', 'Unknown')
                        }
                    )
                    documents.append(doc)
            
            # Process table elements with enhanced metadata
            for i, element in enumerate(extracted_content["table_elements"]):
                content = str(element)
                if content.strip():
                    doc = Document(
                        page_content=f"FINANCIAL TABLE DATA: {content}",
                        metadata={
                            "type": "table",
                            "element_id": i,
                            "category": "Financial_Table",
                            "analysis_type": "tabular_data"
                        }
                    )
                    documents.append(doc)
                    self.extracted_tables.append({
                        "id": i,
                        "content": content,
                        "raw_element": element
                    })
            
            # Process image elements with paths for multimodal analysis
            for i, image_path in enumerate(extracted_content.get("extracted_image_paths", [])):
                if os.path.exists(image_path):
                    # Create description document for image
                    doc = Document(
                        page_content=f"FINANCIAL CHART/GRAPH: Image extracted from document - {os.path.basename(image_path)}",
                        metadata={
                            "type": "image",
                            "element_id": i,
                            "category": "Financial_Visual",
                            "image_path": image_path,
                            "analysis_type": "visual_data"
                        }
                    )
                    documents.append(doc)
                    self.extracted_images.append({
                        "id": i,
                        "path": image_path,
                        "filename": os.path.basename(image_path)
                    })
            
            self.documents = documents
            return len(documents)
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return 0
    
    def create_vectorstore(self):
        """Create FAISS vectorstore from processed documents"""
        try:
            if not self.documents:
                st.error("No documents to process")
                return False
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            split_docs = text_splitter.split_documents(self.documents)
            
            # Create vectorstore
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error creating vectorstore: {str(e)}")
            return False
    
    def get_financial_prompt(self):
        """Get the multimodal financial analysis prompt template"""
        return PromptTemplate(
            template="""You are an advanced multimodal financial analyst assistant specialized in analyzing SEC 10-K reports, financial documents, charts, graphs, and tabular data. 
            You can process and analyze text, images (charts/graphs), and structured table data to provide comprehensive financial insights.
            
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
        """Query documents with multimodal capabilities (text, tables, images)"""
        try:
            if not self.retriever:
                return "Please upload and process a document first."
            
            # Get relevant documents
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            # Separate different types of content
            text_context = []
            table_context = []
            image_analyses = []
            
            for doc in relevant_docs:
                doc_type = doc.metadata.get("type", "text")
                
                if doc_type == "text":
                    text_context.append(doc.page_content)
                elif doc_type == "table":
                    table_context.append(doc.page_content)
                elif doc_type == "image" and "image_path" in doc.metadata:
                    # Analyze image with Gemini multimodal
                    image_path = doc.metadata["image_path"]
                    if os.path.exists(image_path):
                        image_analysis = self.analyze_image_with_gemini(image_path, question)
                        image_analyses.append(f"IMAGE ANALYSIS: {image_analysis}")
            
            # Combine all context types
            combined_context = []
            
            if text_context:
                combined_context.append("TEXT CONTENT:\n" + "\n".join(text_context))
            
            if table_context:
                combined_context.append("TABLE DATA:\n" + "\n".join(table_context))
            
            if image_analyses:
                combined_context.append("VISUAL ANALYSIS:\n" + "\n".join(image_analyses))
            
            full_context = "\n\n".join(combined_context)
            
            # Create enhanced retrieval chain
            prompt = self.get_financial_prompt()
            
            # Get chat history
            chat_history = self.memory.chat_memory.messages
            chat_history_str = ""
            for msg in chat_history[-6:]:  # Last 6 messages for context
                if hasattr(msg, 'content'):
                    chat_history_str += f"{msg.__class__.__name__}: {msg.content}\n"
            
            # Generate response using the multimodal context
            response = self.llm.invoke(
                prompt.format(
                    context=full_context,
                    chat_history=chat_history_str,
                    input=question
                )
            )
            
            # Save to memory
            self.memory.save_context({"input": question}, {"output": response.content})
            
            return response.content
            
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return "Sorry, I encountered an error while processing your question."

# Initialize session state
if "rag_processor" not in st.session_state:
    st.session_state.rag_processor = FinancialRAGProcessor()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

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
    
    if google_api_key:
        if st.session_state.rag_processor.llm is None:
            with st.spinner("Initializing models..."):
                success = st.session_state.rag_processor.initialize_models(google_api_key)
                if success:
                    st.success("✅ Models initialized!")
                else:
                    st.error("❌ Failed to initialize models")
    
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload Financial Document (PDF)",
        type=['pdf'],
        help="Upload SEC 10-K reports or other financial documents"
    )
    
    if uploaded_file and google_api_key:
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner("Extracting content from PDF..."):
                # Extract content
                extracted_content = st.session_state.rag_processor.extract_pdf_content(uploaded_file)
                
                if extracted_content:
                    st.success(f"✅ Extracted {extracted_content['total_elements']} elements")
                    
                    # Process documents
                    with st.spinner("Processing documents..."):
                        doc_count = st.session_state.rag_processor.process_documents(extracted_content)
                        st.success(f"✅ Processed {doc_count} document chunks")
                    
                    # Create vectorstore
                    with st.spinner("Creating knowledge base..."):
                        if st.session_state.rag_processor.create_vectorstore():
                            st.success("✅ Knowledge base created!")
                            st.session_state.documents_processed = True
                        else:
                            st.error("❌ Failed to create knowledge base")
    
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

# Main content area
if not google_api_key:
    st.info("👈 Please enter your Google API key in the sidebar to get started.")
elif not st.session_state.documents_processed:
    st.info("👈 Please upload and process a financial document to start analyzing.")
else:
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
    
    cols = st.columns(3)  # Changed from len(sample_questions) to 3 for better layout
    for i, question in enumerate(sample_questions):
        col_idx = i % 3
        with cols[col_idx]:
            if st.button(question, key=f"sample_{i}"):
                st.session_state.current_question = question
    
    # Chat interface
    st.header("💬 Financial Analysis Chat")
    
    # Display chat history
    for message in st.session_state.chat_history:
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
        with st.spinner("Analyzing document..."):
            response = st.session_state.rag_processor.query_documents(query)
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
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
        st.session_state.chat_history = []
        if st.session_state.rag_processor.memory:
            st.session_state.rag_processor.memory.clear()
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
