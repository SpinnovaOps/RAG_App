import os
import streamlit as st # type: ignore
from dotenv import load_dotenv # type: ignore
import pandas as pd
import time

from document_processor import DocumentProcessor # type: ignore
from rag_engine import RAGEngine # type: ignore

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY not found in environment variables. Please set it in a .env file.")
    st.stop()

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'current_document_id' not in st.session_state:
    st.session_state.current_document_id = None
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'suggested_questions' not in st.session_state:
    st.session_state.suggested_questions = []

# Initialize processor
processor = DocumentProcessor(upload_folder="uploads")

# Set page configuration
st.set_page_config(
    page_title="SimpleGeminiRAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Functions for document handling
def upload_document():
    """Handle document upload and processing."""
    uploaded_file = st.session_state.uploaded_file
    
    if uploaded_file is not None:
        # Save the file temporarily
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                # Process the document
                document = processor.process_document(file_path, uploaded_file.name)
                
                # Generate embeddings
                embeddings, vectorizer = processor.generate_embeddings(document.chunks)
                
                # Add to RAG engine
                st.session_state.rag_engine.add_document(document, embeddings, vectorizer)
                
                # Store in session state
                st.session_state.documents[document.id] = {
                    "id": document.id,
                    "filename": document.filename,
                    "chunk_count": len(document.chunks)
                }
                
                # Set as current document
                st.session_state.current_document_id = document.id
                
                # Reset chat history
                st.session_state.chat_history = []
                
                # Generate suggested questions
                generate_suggested_questions(document.id)
                
                st.success(f"Successfully processed {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

def select_document():
    """Handle document selection."""
    st.session_state.current_document_id = st.session_state.selected_document
    st.session_state.chat_history = []
    
    # Generate suggested questions for the selected document
    generate_suggested_questions(st.session_state.current_document_id)

def generate_suggested_questions(document_id):
    """Generate suggested questions for a document."""
    with st.spinner("Generating suggested questions..."):
        try:
            questions = st.session_state.rag_engine.generate_questions(document_id)
            st.session_state.suggested_questions = questions
        except Exception as e:
            st.error(f"Error generating questions: {str(e)}")
            st.session_state.suggested_questions = []

def handle_chat_input():
    """Process user query and generate response."""
    query = st.session_state.user_query
    
    if not query:
        return
    
    if not st.session_state.current_document_id:
        st.warning("Please upload or select a document first")
        return
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    # Clear input
    st.session_state.user_query = ""
    
    with st.spinner("Searching document and generating response..."):
        try:
            # Search for relevant chunks
            results = st.session_state.rag_engine.search(
                query, 
                st.session_state.current_document_id,
                top_k=5
            )
            
            if not results:
                response = "I couldn't find relevant information in the document to answer your question."
            else:
                # Generate answer
                response, _ = st.session_state.rag_engine.generate_answer(query, results)
            
            # Add sources information
            sources_info = ""
            if results:
                sources_info = "\n\n**Sources:**\n"
                for i, (chunk, score) in enumerate(results):
                    # Format as markdown with confidence score
                    sources_info += f"- Chunk {chunk.metadata.get('position', 'unknown')} (Score: {score:.2f})\n"
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response + sources_info
            })
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"An error occurred: {str(e)}"
            })

def use_suggested_question(question):
    """Use a suggested question as user input."""
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    with st.spinner("Searching document and generating response..."):
        try:
            # Search for relevant chunks
            results = st.session_state.rag_engine.search(
                question, 
                st.session_state.current_document_id,
                top_k=5
            )
            
            if not results:
                response = "I couldn't find relevant information in the document to answer your question."
            else:
                # Generate answer
                response, _ = st.session_state.rag_engine.generate_answer(question, results)
            
            # Add sources information
            sources_info = ""
            if results:
                sources_info = "\n\n**Sources:**\n"
                for i, (chunk, score) in enumerate(results):
                    # Format as markdown with confidence score
                    sources_info += f"- Chunk {chunk.metadata.get('position', 'unknown')} (Score: {score:.2f})\n"
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response + sources_info
            })
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"An error occurred: {str(e)}"
            })

def generate_document_summary():
    """Generate a summary of the current document."""
    if not st.session_state.current_document_id:
        st.warning("Please select a document first")
        return
    
    with st.spinner("Generating document summary..."):
        try:
            summary_length = st.session_state.get("summary_length", "medium")
            summary, _ = st.session_state.rag_engine.generate_summary(
                st.session_state.current_document_id, 
                length=summary_length
            )
            
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "system", 
                "content": f"**Document Summary ({summary_length}):**\n\n{summary}"
            })
            
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")

def analyze_current_document():
    """Run a comprehensive analysis on the current document."""
    if not st.session_state.current_document_id:
        st.warning("Please select a document first")
        return
    
    with st.spinner("Analyzing document..."):
        try:
            analysis = st.session_state.rag_engine.analyze_document(
                st.session_state.current_document_id
            )
            
            # Format analysis results
            metadata = analysis.get("metadata", {})
            summary = analysis.get("summary", "No summary available")
            questions = analysis.get("suggested_questions", [])
            
            # Add to chat history
            content = f"""**Document Analysis Report**

**Document**: {metadata.get('title', 'Untitled')} (ID: {metadata.get('id', 'unknown')})
**Word Count**: {metadata.get('word_count', 'unknown')}  
**Chunks**: {metadata.get('chunk_count', 'unknown')}
**Token Estimate**: {metadata.get('estimated_token_count', 'unknown')}

**Summary**:
{summary}

**Key Questions**:
"""
            for i, q in enumerate(questions):
                content += f"{i+1}. {q}\n"
            
            st.session_state.chat_history.append({
                "role": "system", 
                "content": content
            })
            
        except Exception as e:
            st.error(f"Error analyzing document: {str(e)}")

# Main UI
st.title("SimpleGeminiRAG 📚")
st.subheader("Document Q&A with Gemini and RAG")

# Sidebar for document management
with st.sidebar:
    st.header("Document Management")
    
    # Upload new document
    st.subheader("Upload a Document")
    st.file_uploader(
        "Choose a PDF, DOCX, or TXT file", 
        type=["pdf", "docx", "txt"], 
        key="uploaded_file",
        on_change=upload_document
    )
    
    # Select existing document
    if st.session_state.documents:
        st.subheader("Select a Document")
        doc_options = {doc["filename"]: doc["id"] for doc in st.session_state.documents.values()}
        st.selectbox(
            "Choose a document", 
            options=list(doc_options.keys()),
            key="selected_document",
            on_change=select_document,
            format_func=lambda x: x
        )
        
        # Document actions
        if st.session_state.current_document_id:
            st.subheader("Document Actions")
            
            # Summary options
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox(
                    "Summary Length", 
                    options=["short", "medium", "long"],
                    key="summary_length",
                    index=1
                )
            with col2:
                st.button("Generate Summary", on_click=generate_document_summary)
            
            # Analysis
            st.button("Full Document Analysis", on_click=analyze_current_document)

# Main content area
main_col, side_col = st.columns([3, 1])

with main_col:
    # Chat interface
    st.subheader("Document Q&A")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.markdown(f"**You**: {content}")
            elif role == "assistant":
                st.markdown(f"**Assistant**: {content}")
            else:  # system messages
                st.markdown(content)
    
    # Input for new questions
    st.text_input(
        "Ask a question about the document", 
        key="user_query",
        on_change=handle_chat_input,
        disabled=not st.session_state.current_document_id
    )
    
    if not st.session_state.current_document_id:
        st.info("Please upload or select a document to start asking questions")

with side_col:
    # Display document info and suggested questions
    if st.session_state.current_document_id:
        doc_info = st.session_state.documents.get(st.session_state.current_document_id)
        if doc_info:
            st.subheader("Current Document")
            st.markdown(f"**File**: {doc_info['filename']}")
            st.markdown(f"**Chunks**: {doc_info['chunk_count']}")
            
            # Suggested questions
            if st.session_state.suggested_questions:
                st.subheader("Suggested Questions")
                for question in st.session_state.suggested_questions:
                    if st.button(question, key=f"btn_{hash(question)}"):
                        use_suggested_question(question)
    else:
        st.info("No document selected")

# Footer
st.markdown("---")
st.caption("SimpleGeminiRAG: Powered by Google Gemini and Streamlit")