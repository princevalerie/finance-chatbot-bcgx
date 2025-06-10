import os
# Nonaktifkan file watcher Streamlit untuk menghindari konflik dengan PyTorch
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import torch
# Patch PyTorch path handling
torch.classes.__path__ = []

import asyncio
from typing import List, Dict, Any
import tempfile
from pathlib import Path
import requests
import re
from urllib.parse import urlparse, urljoin
import time

# PDF conversion imports
# import pdfkit # Dihapus
import io
from weasyprint import HTML # Menambahkan import weasyprint

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

# Other imports
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

class SECDocumentProcessor:
    """Class for processing SEC documents from URLs"""
    
    def __init__(self):
        # self.wkhtmltopdf_options = { # Dihapus karena tidak lagi diperlukan
        #     'page-size': 'A4',
        #     'margin-top': '0.75in',
        #     'margin-right': '0.75in',
        #     'margin-bottom': '0.75in',
        #     'margin-left': '0.75in',
        #     'encoding': "UTF-8",
        #     'no-outline': None,
        #     'enable-local-file-access': None
        # }
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.memory = None
        self.conversation_chain = None
    
    def validate_sec_url(self, url: str) -> bool:
        """Validate if URL is from SEC website"""
        parsed_url = urlparse(url)
        return parsed_url.netloc in ['www.sec.gov', 'sec.gov']
    
    def extract_company_info(self, url: str) -> Dict[str, str]:
        """Extract company information from SEC URL"""
        try:
            # Extract CIK and document info from URL pattern
            cik_match = re.search(r'/data/(\d+)/', url)
            filename_match = re.search(r'/([^/]+\.htm?)', url)
            
            if cik_match and filename_match:
                return {
                    'cik': cik_match.group(1),
                    'filename': filename_match.group(1)
                }
            return None
        except Exception as e:
            st.error(f"Error extracting company info: {str(e)}")
            return None

    def get_document_metadata(self, url: str) -> Dict[str, Any]:
        """Get metadata from SEC document URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.head(url, headers=headers, allow_redirects=True)
            
            metadata = {
                'url': url,
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', 'Unknown'),
                'content_length': response.headers.get('content-length', 'Unknown'),
                'last_modified': response.headers.get('last-modified', 'Unknown')
            }
            
            return metadata
            
        except Exception as e:
            st.error(f"Error getting document metadata: {str(e)}")
            return {
                'url': url,
                'status_code': 'Error',
                'content_type': 'Unknown',
                'content_length': 'Unknown',
                'last_modified': 'Unknown'
            }

    def convert_url_to_pdf(self, url: str) -> str:
        """Convert SEC document URL to PDF and save to a temporary file using WeasyPrint"""
        try:
            # Headers khusus untuk SEC.gov
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Referer': 'https://www.sec.gov/',
                'Origin': 'https://www.sec.gov'
            }
            
            # Buat session untuk menjaga cookies
            session = requests.Session()
            
            # Tambahkan delay untuk menghindari rate limiting
            time.sleep(3)
            
            # Coba download dengan session dan timeout yang lebih lama
            response = session.get(
                url, 
                headers=headers, 
                timeout=60,
                allow_redirects=True
            )
            
            # Cek status code
            if response.status_code == 403:
                st.error("Access denied by SEC.gov. Possible solutions:\n"
                        "1. URL mungkin memerlukan autentikasi\n"
                        "2. Dokumen mungkin tidak tersedia untuk umum\n"
                        "3. Coba akses melalui browser terlebih dahulu")
                raise requests.exceptions.HTTPError("403 Forbidden")
                
            response.raise_for_status()
            
            # Tambahkan delay sebelum konversi
            time.sleep(2)
            
            # Gunakan HTML content untuk WeasyPrint
            htmldoc = HTML(string=response.text)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf_file:
                htmldoc.write_pdf(tmp_pdf_file.name)
                pdf_path = tmp_pdf_file.name
            
            st.success(f"Document converted and saved to {pdf_path}")
            return pdf_path
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                st.error("Access denied by SEC.gov. Please try:\n"
                        "1. Verifikasi URL di browser\n"
                        "2. Pastikan dokumen tersedia untuk umum\n"
                        "3. Tunggu beberapa menit dan coba lagi")
            else:
                st.error(f"HTTP Error: {str(e)}")
            raise
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
            raise
        except Exception as e:
            st.error(f"Failed to convert URL to PDF: {str(e)}")
            raise

    def initialize_embeddings(self):
        """Initialize Qwen embeddings from Hugging Face"""
        try:
            # Using exact model from the image: Qwen/Qwen3-Embedding-0.6B
            model_name = "Qwen/Qwen3-Embedding-0.6B"
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True  # Required for Qwen models
                },
                encode_kwargs={'normalize_embeddings': True}
            )
            st.success(f"✅ Qwen3-Embedding-0.6B initialized successfully")
            return True
        except Exception as e:
            st.error(f"Error initializing Qwen embeddings: {str(e)}")
            # Fallback to sentence-transformers if Qwen fails
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                st.warning("Using fallback embedding model: all-MiniLM-L6-v2")
                return True
            except Exception as fallback_e:
                st.error(f"Fallback embedding also failed: {str(fallback_e)}")
                return False
    
    def initialize_llm(self, api_key: str):
        """Initialize Gemini LLM"""
        try:
            genai.configure(api_key=api_key)
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=api_key,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            return True
        except Exception as e:
            st.error(f"Error initializing Gemini LLM: {str(e)}")
            return False
    
    def process_pdf_with_unstructured(self, pdf_file) -> List[Document]:
        """Process PDF using Unstructured library"""
        documents = []
        
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Process PDF with unstructured
            elements = partition_pdf(
                filename=tmp_file_path,
                strategy="hi_res",  # High resolution for better text extraction
                infer_table_structure=True,  # Important for financial documents
                extract_images_in_pdf=True,
                chunking_strategy="by_title"
            )
            
            # Convert elements to documents
            for element in elements:
                if hasattr(element, 'text') and element.text.strip():
                    # Add metadata for better context
                    metadata = {
                        'source': pdf_file.name,
                        'element_type': str(type(element).__name__),
                        'page_number': getattr(element, 'metadata', {}).get('page_number', 'unknown')
                    }
                    
                    documents.append(Document(
                        page_content=element.text,
                        metadata=metadata
                    ))
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return documents
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return []
    
    def create_vectorstore(self, documents: List[Document]):
        """Create FAISS vectorstore from documents"""
        if not documents:
            st.error("No documents to process")
            return False
        
        try:
            # Split documents into smaller chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            split_documents = text_splitter.split_documents(documents)
            
            # Create vectorstore
            self.vectorstore = FAISS.from_documents(
                documents=split_documents,
                embedding=self.embeddings
            )
            
            st.success(f"Created vectorstore with {len(split_documents)} document chunks")
            return True
            
        except Exception as e:
            st.error(f"Error creating vectorstore: {str(e)}")
            return False
    
    def setup_conversation_chain(self):
        """Setup conversational retrieval chain with memory"""
        if not self.vectorstore or not self.llm:
            return False
        
        try:
            # Create memory for conversation history
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Custom prompt for financial analysis
            financial_prompt = PromptTemplate(
                template="""You are a financial analyst assistant specialized in analyzing SEC 10-K reports and financial documents. 
                Use the following context to answer questions about financial data, company performance, risks, and market analysis.
                
                Context: {context}
                
                Chat History: {chat_history}
                
                Question: {question}
                
                Provide detailed, accurate financial analysis based on the document context. Include specific numbers, ratios, and trends when available.
                If you cannot find the specific information in the context, clearly state that the information is not available in the provided documents.
                
                Answer:""",
                input_variables=["context", "chat_history", "question"]
            )
            
            # Create history aware retriever
            history_aware_retriever = create_history_aware_retriever(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                ),
                rephraser_prompt=financial_prompt
            )
            
            # Create document chain
            document_chain = create_stuff_documents_chain(
                llm=self.llm,
                prompt=financial_prompt
            )
            
            # Create retrieval chain
            self.conversation_chain = create_retrieval_chain(
                retriever=history_aware_retriever,
                combine_docs_chain=document_chain
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error setting up conversation chain: {str(e)}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the chatbot"""
        if not self.conversation_chain:
            return {"error": "Chatbot not properly initialized"}
        
        try:
            response = self.conversation_chain.invoke({
                "question": question,
                "chat_history": self.memory.chat_memory.messages if self.memory else []
            })
            
            # Update memory
            if self.memory:
                self.memory.save_context(
                    {"input": question},
                    {"output": response["answer"]}
                )
            
            return {
                "answer": response["answer"],
                "source_documents": response.get("source_documents", [])
            }
        except Exception as e:
            return {"error": f"Error processing query: {str(e)}"}

# Wrapper class for the Streamlit UI
class FinancialRAGChatbot:
    """Class untuk chatbot RAG khusus analisis finansial yang membungkus SECDocumentProcessor"""
    
    def __init__(self):
        self.processor = SECDocumentProcessor()
        
    def initialize_embeddings(self):
        return self.processor.initialize_embeddings()
    
    def initialize_llm(self, api_key: str):
        return self.processor.initialize_llm(api_key)
        
    def process_pdf_with_unstructured(self, pdf_file) -> List[Document]:
        return self.processor.process_pdf_with_unstructured(pdf_file)
    
    def create_vectorstore(self, documents: List[Document]):
        return self.processor.create_vectorstore(documents)
    
    def setup_conversation_chain(self):
        return self.processor.setup_conversation_chain()
    
    def query(self, question: str) -> Dict[str, Any]:
        return self.processor.query(question)

def sec_url_input_page():
    """Page for SEC URL input and conversion"""
    st.set_page_config(
        page_title="SEC Document Processor",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 SEC Document Processor")
    st.markdown("*Convert SEC documents to PDF for financial analysis*")
    
    # Initialize SEC processor
    if 'sec_processor' not in st.session_state:
        st.session_state.sec_processor = SECDocumentProcessor()
    
    # URL input section
    st.header("🔗 Enter SEC Document URL")
    
    # Example URLs for reference
    with st.expander("📋 Example SEC URLs"):
        st.markdown("""
        **Tesla 10-K Report:**
        ```
        https://www.sec.gov/Archives/edgar/data/1318605/000156459020047486/tsla-ex101_69.htm
        ```
        
        **Apple 10-K Report:**
        ```
        https://www.sec.gov/Archives/edgar/data/320193/000032019323000077/aapl-20230930.htm
        ```
        
        **Microsoft 10-K Report:**
        ```
        https://www.sec.gov/Archives/edgar/data/789019/000156459023034948/msft-ex101_6.htm
        ```
        """)
    
    # URL input
    sec_url = st.text_input(
        "SEC Document URL:",
        placeholder="https://www.sec.gov/Archives/edgar/data/...",
        help="Enter the full URL of the SEC document you want to analyze"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔍 Validate URL", disabled=not sec_url):
            if sec_url:
                processor = st.session_state.sec_processor
                
                if processor.validate_sec_url(sec_url):
                    st.success("✅ Valid SEC URL")
                    
                    # Get document metadata
                    with st.spinner("Getting document information..."):
                        metadata = processor.get_document_metadata(sec_url)
                        company_info = processor.extract_company_info(sec_url)
                    
                    # Display document info
                    st.subheader("📊 Document Information")
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.write(f"**CIK:** {company_info['cik']}")
                        st.write(f"**Filename:** {company_info['filename']}")
                        st.write(f"**Status:** {metadata.get('status_code', 'Unknown')}")
                    
                    with col_info2:
                        st.write(f"**Content Type:** {metadata.get('content_type', 'Unknown')}")
                        st.write(f"**Size:** {metadata.get('content_length', 'Unknown')}")
                        st.write(f"**Last Modified:** {metadata.get('last_modified', 'Unknown')}")
                    
                    st.session_state.validated_url = sec_url
                    st.session_state.document_info = {**company_info, **metadata}
                    
                else:
                    st.error("❌ Invalid URL. Please enter a valid SEC document URL.")
    
    with col2:
        if st.button("📄 Convert to PDF & Analyze", disabled=not hasattr(st.session_state, 'validated_url')):
            if hasattr(st.session_state, 'validated_url'):
                try:
                    processor = st.session_state.sec_processor
                    
                    with st.spinner("Converting SEC document to PDF..."):
                        # Convert URL to PDF (menggunakan asyncio.run untuk memanggil fungsi async)
                        pdf_path = asyncio.run(processor.convert_url_to_pdf(st.session_state.validated_url))
                        st.session_state.converted_pdf_path = pdf_path
                        st.session_state.show_chatbot = True
                    
                    st.success("✅ Document converted successfully!")
                    st.info("🚀 Redirecting to analysis interface...")
                    
                    # Small delay before rerun
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Conversion failed: {str(e)}")
    
    # Installation instructions
    st.divider()
    st.header("⚙️ Setup Instructions")
    
    with st.expander("📦 Required Dependencies"):
        st.markdown("""
        **Instal WeasyPrint dependencies (pastikan memiliki paket-paket yang diperlukan seperti pango, cairo, dll.):**
        ```bash
        # Contoh di Ubuntu/Debian:
        sudo apt-get install libpango1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libcairo2 libffi-dev
        # Untuk sistem lain, lihat dokumentasi WeasyPrint: https://doc.courtbouillon.org/weasyprint/stable/install.html
        ```
        
        **Python packages:**
        ```bash
        pip install -r requirements.txt
        ```
        """)

def main():
    # Check if we should show chatbot or URL input
    if not hasattr(st.session_state, 'show_chatbot') or not st.session_state.show_chatbot:
        sec_url_input_page()
        return
    
    # Main chatbot interface
    st.set_page_config(
        page_title="Financial RAG Chatbot",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Financial Analysis RAG Chatbot")
    st.markdown("*Analyzing SEC document with AI-powered insights*")
    
    # Show document info
    if hasattr(st.session_state, 'document_info'):
        with st.expander("📄 Document Information", expanded=False):
            doc_info = st.session_state.document_info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**CIK:** {doc_info.get('cik', 'N/A')}")
                st.write(f"**Filename:** {doc_info.get('filename', 'N/A')}")
            with col2:
                st.write(f"**URL:** {doc_info.get('url', 'N/A')}")
                st.write(f"**Status:** {doc_info.get('status_code', 'N/A')}")
            with col3:
                if st.button("🔄 Process New Document"):
                    # Reset session state
                    for key in ['show_chatbot', 'converted_pdf_path', 'document_info', 'validated_url']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = FinancialRAGChatbot()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vectorstore_ready' not in st.session_state:
        st.session_state.vectorstore_ready = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Gemini API Key
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        
        if gemini_api_key:
            if st.session_state.chatbot.initialize_llm(gemini_api_key):
                st.success("✅ Gemini LLM initialized")
            
        # Initialize embeddings
        if st.button("Initialize Embeddings"):
            with st.spinner("Loading Qwen embeddings..."):
                if st.session_state.chatbot.initialize_embeddings():
                    st.success("✅ Embeddings initialized")
        
        st.divider()
        
        # File upload
        st.header("📄 Document Processing")
        
        # Auto-load converted PDF if available
        if hasattr(st.session_state, 'converted_pdf_path') and os.path.exists(st.session_state.converted_pdf_path):
            st.info(f"📄 Converted PDF ready: {os.path.basename(st.session_state.converted_pdf_path)}")
            
            if st.button("🚀 Process Converted Document") and st.session_state.chatbot.processor.embeddings:
                with st.spinner("Processing converted SEC document..."):
                    # Read the converted PDF
                    with open(st.session_state.converted_pdf_path, 'rb') as f:
                        pdf_data = f.read()
                    
                    # Create a file-like object
                    class PDFFileObj:
                        def __init__(self, data, name):
                            self.data = data
                            self.name = name
                        def getvalue(self):
                            return self.data
                    
                    pdf_file = PDFFileObj(pdf_data, os.path.basename(st.session_state.converted_pdf_path))
                    
                    # Process the PDF
                    documents = st.session_state.chatbot.process_pdf_with_unstructured(pdf_file)
                    
                    if documents:
                        if st.session_state.chatbot.create_vectorstore(documents):
                            if st.session_state.chatbot.setup_conversation_chain():
                                st.session_state.vectorstore_ready = True
                                st.success("✅ SEC document processed and ready for analysis!")
                            else:
                                st.error("Failed to setup conversation chain")
                        else:
                            st.error("Failed to create vectorstore")
                    else:
                        st.error("No content extracted from document")
        else:
            st.info("Upload additional PDF files if needed")
        
        uploaded_files = st.file_uploader(
            "Upload additional PDF files (optional)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload additional SEC reports or financial documents"
        )
        
        if uploaded_files and st.session_state.chatbot.processor.embeddings:
            if st.button("Process Documents"):
                all_documents = []
                
                progress_bar = st.progress(0)
                for i, uploaded_file in enumerate(uploaded_files):
                    st.info(f"Processing {uploaded_file.name}...")
                    
                    documents = st.session_state.chatbot.process_pdf_with_unstructured(uploaded_file)
                    all_documents.extend(documents)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                if all_documents:
                    st.info("Creating vectorstore...")
                    if st.session_state.chatbot.create_vectorstore(all_documents):
                        if st.session_state.chatbot.setup_conversation_chain():
                            st.session_state.vectorstore_ready = True
                            st.success("✅ Documents processed and ready for analysis!")
                        else:
                            st.error("Failed to setup conversation chain")
                    else:
                        st.error("Failed to create vectorstore")
                else:
                    st.error("No documents were successfully processed")
        
        # Sample questions
        st.divider()
        st.header("💡 Sample Questions")
        sample_questions = [
            "What are the key financial metrics for this company?",
            "What are the main risk factors mentioned?",
            "How did revenue change compared to previous year?",
            "What is the company's debt-to-equity ratio?",
            "What are the major business segments?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question[:20]}"):
                if st.session_state.vectorstore_ready:
                    st.session_state.chat_history.append({"role": "user", "content": question})
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                        if "sources" in message:
                            with st.expander("📚 Sources"):
                                for i, source in enumerate(message["sources"]):
                                    st.write(f"**Source {i+1}:** {source}")
        
        # Chat input
        if st.session_state.vectorstore_ready:
            user_question = st.chat_input("Ask about the financial documents...")
            
            if user_question:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                # Get response from chatbot
                with st.spinner("Analyzing documents..."):
                    response = st.session_state.chatbot.query(user_question)
                
                if "error" in response:
                    st.error(response["error"])
                else:
                    # Add assistant response to history
                    sources = []
                    if "source_documents" in response:
                        sources = [doc.page_content[:200] + "..." for doc in response["source_documents"]]
                    
                    assistant_message = {
                        "role": "assistant", 
                        "content": response["answer"],
                        "sources": sources
                    }
                    st.session_state.chat_history.append(assistant_message)
                
                st.rerun()
        else:
            st.info("Please configure the API key, initialize embeddings, and upload documents to start chatting.")
    
    with col2:
        st.header("📈 Analysis Tools")
        
        if st.session_state.vectorstore_ready:
            st.success("✅ Ready for financial analysis")
            
            # Quick analysis buttons
            if st.button("📊 Financial Summary"):
                summary_query = "Provide a comprehensive financial summary including revenue, profit margins, and key financial ratios."
                st.session_state.chat_history.append({"role": "user", "content": summary_query})
                st.rerun()
            
            if st.button("⚠️ Risk Analysis"):
                risk_query = "What are the main risk factors and challenges facing this company?"
                st.session_state.chat_history.append({"role": "user", "content": risk_query})
                st.rerun()
            
            if st.button("📈 Performance Trends"):
                trend_query = "Analyze the company's performance trends over the reporting periods."
                st.session_state.chat_history.append({"role": "user", "content": trend_query})
                st.rerun()
        else:
            st.warning("Complete setup to enable analysis tools")
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()
    
