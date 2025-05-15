import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import streamlit as st

# It's good practice to load API key from environment variables
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     st.error("GOOGLE_API_KEY environment variable not set.")
# else:
#     genai.configure(api_key=GOOGLE_API_KEY)

# Placeholder for where vector store will be saved/loaded from
VECTOR_STORE_PATH = "vector_store/event_attendees_faiss"
RESUMES_DIR_NAME = "resumes" # Directory to scan for resumes

def test_embedding_model(api_key):
    """Tests if the embedding model can be initialized and used with the API key."""
    if not api_key:
        st.error("API Key not provided for embedding test.")
        print("[DEBUG_EMBED_TEST] API Key not provided.")
        return False
    try:
        print("[DEBUG_EMBED_TEST] Configuring genai with API key...")
        genai.configure(api_key=api_key)
        print("[DEBUG_EMBED_TEST] Initializing GoogleGenerativeAIEmbeddings with explicit API key...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        print("[DEBUG_EMBED_TEST] Attempting to embed a test string...")
        test_vector = embeddings.embed_query("This is a test string.")
        if test_vector and isinstance(test_vector, list) and len(test_vector) > 0:
            print(f"[DEBUG_EMBED_TEST] Successfully embedded test string. Vector length: {len(test_vector)}")
            st.success("Embedding model test successful.") # Provide feedback in UI
            return True
        else:
            st.error("Embedding model test failed: Did not receive a valid vector.")
            print("[DEBUG_EMBED_TEST] Embedding test failed: Did not receive a valid vector.")
            return False
    except Exception as e:
        st.error(f"Embedding model test failed: {e}")
        print(f"[DEBUG_EMBED_TEST] Exception during embedding test: {e}", flush=True)
        return False

def load_document(file_path):
    """Loads a document (PDF or DOCX) from the given file path."""
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            print(f"[DEBUG] Loading PDF: {file_path}")
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            print(f"[DEBUG] Loading DOCX: {file_path}")
        else:
            st.warning(f"Unsupported file type: {file_path}. Skipping.")
            return None
            
        docs = loader.load()
        print(f"[DEBUG] Loaded {len(docs)} pages/sections from {file_path}")
        if not docs:
            st.warning(f"No content extracted from {file_path}")
            return None
            
        # Add more metadata to help with retrieval
        for doc in docs:
            doc.metadata.update({
                "source": os.path.basename(file_path),
                "file_path": file_path,
                "file_type": file_path.split(".")[-1].lower(),
                "page_count": len(docs)
            })
        return docs
    except Exception as e:
        st.error(f"Error loading document {file_path}: {e}")
        print(f"[DEBUG] Document loading error for {file_path}: {str(e)}")
        return None

def get_text_chunks(documents):
    """Splits documents into text chunks."""
    if not documents:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Reduced chunk size for more granular extraction
        chunk_overlap=100,  # Adjusted overlap
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Added more separators for better chunking
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[DEBUG] Created {len(chunks)} chunks from documents")
    return chunks

def get_vector_store(text_chunks, api_key):
    """Creates a FAISS vector store from text chunks using Google GenAI embeddings."""
    if not text_chunks:
        st.warning("No text chunks to process for vector store.")
        return None
    try:
        if not api_key:
            st.error("Google API Key is not configured. Cannot create embeddings.")
            return None
        
        print(f"[DEBUG] Initializing vector store with {len(text_chunks)} chunks")
        genai.configure(api_key=api_key)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
            task_type="retrieval_document"  # Explicitly set task type for document retrieval
        )
        
        # Check if a vector store already exists
        if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH): # Check if directory is not empty
            try:
                st.info(f"Loading existing vector store from {VECTOR_STORE_PATH}")
                vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
                st.info("Adding new documents to existing vector store...")
                
                # Create embeddings in smaller batches to avoid potential memory issues
                batch_size = 50
                for i in range(0, len(text_chunks), batch_size):
                    batch = text_chunks[i:i + batch_size]
                    print(f"[DEBUG] Processing batch {i//batch_size + 1} of {(len(text_chunks)-1)//batch_size + 1}")
                    vector_store.add_documents(batch)
                
                vector_store.save_local(VECTOR_STORE_PATH)
                st.success("Successfully updated vector store.")
                return vector_store
            except Exception as e:
                st.warning(f"Could not load existing vector store or add documents: {e}. Creating a new one.")
                print(f"[DEBUG] Vector store loading/updating error: {str(e)}")
        
        st.info("Creating new vector store...")
        print(f"[DEBUG] Creating new vector store from {len(text_chunks)} chunks")
        
        # Create vector store in batches
        batch_size = 50
        first_batch = text_chunks[:batch_size]
        vector_store = FAISS.from_documents(documents=first_batch, embedding=embeddings)
        
        # Process remaining chunks in batches
        for i in range(batch_size, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            print(f"[DEBUG] Processing batch {i//batch_size + 1} of {(len(text_chunks)-1)//batch_size + 1}")
            vector_store.add_documents(batch)
        
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        vector_store.save_local(VECTOR_STORE_PATH)
        st.success(f"Vector store created and saved to {VECTOR_STORE_PATH}")
        return vector_store

    except Exception as e:
        print(f"[DEBUG] Exception in get_vector_store: {e}", flush=True)
        st.error(f"Error creating vector store: {e}")
        return None

def process_resumes_from_folder(api_key):
    """Scans the RESUMES_DIR_NAME for PDF and DOCX files, processes them, and updates/creates a vector store."""
    if not api_key:
        st.error("Google API Key is not provided. Cannot process resumes.")
        return None

    # --- Test embedding model connectivity first ---
    print("[DEBUG] Running embedding model test before processing resumes...")
    if not test_embedding_model(api_key):
        st.error("Embedding model test failed. Cannot proceed with resume processing.")
        # Optionally, you might want to clear any success message from the test that might have appeared before the final st.error here.
        return None 
    # --- End embedding model test ---

    if not os.path.exists(RESUMES_DIR_NAME):
        st.warning(f"Resumes directory '{RESUMES_DIR_NAME}' not found. Please create it and add resumes.")
        return None

    processed_docs = []
    files_in_resumes_dir = os.listdir(RESUMES_DIR_NAME)
    resume_files_to_process = [
        f for f in files_in_resumes_dir 
        if os.path.isfile(os.path.join(RESUMES_DIR_NAME, f)) and (f.lower().endswith(".pdf") or f.lower().endswith(".docx"))
    ]

    if not resume_files_to_process:
        st.info(f"No PDF or DOCX resume files found in the '{RESUMES_DIR_NAME}' directory.")
        return None

    st.info(f"Found {len(resume_files_to_process)} resume(s) in '{RESUMES_DIR_NAME}'. Processing...")

    for filename in resume_files_to_process:
        file_path = os.path.join(RESUMES_DIR_NAME, filename)
        try:
            # st.write(f"Processing {filename}...") # Can be verbose, use st.info for summary
            doc_content = load_document(file_path)
            if doc_content:
                for doc_chunk in doc_content:
                    doc_chunk.metadata["source"] = filename # Use filename as source
                processed_docs.extend(doc_content)
            else:
                st.warning(f"Could not load document: {filename}")
        except Exception as e:
            st.error(f"Error processing file {filename}: {e}")

    if not processed_docs:
        st.warning("No resume documents were successfully processed from the folder.")
        return None

    text_chunks = get_text_chunks(processed_docs)
    if not text_chunks:
        print("[DEBUG] No text chunks generated in process_resumes_from_folder.")
        st.warning("No text chunks generated from resumes in the folder.")
        return None
        
    print("[DEBUG] Calling get_vector_store from process_resumes_from_folder.")
    vector_store = get_vector_store(text_chunks, api_key)
    
    if vector_store:
        st.success(f"Resumes from '{RESUMES_DIR_NAME}' processed. Vector store updated/created.")
        return vector_store
    else:
        st.error(f"Failed to create/update vector store from resumes in '{RESUMES_DIR_NAME}'.")
        return None

def get_existing_vector_store(api_key):
    """Loads an existing vector store if available."""
    if not api_key:
        st.error("Google API Key is not provided. Cannot load vector store.")
        return None

    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        try:
            genai.configure(api_key=api_key)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            st.info(f"Loading existing vector store from {VECTOR_STORE_PATH}")
            vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            st.success("Successfully loaded vector store.")
            return vector_store
        except Exception as e:
            st.error(f"Error loading existing vector store: {e}. You might need to upload resumes first.")
            return None
    else:
        st.info("No existing vector store found. Please upload resumes to create one.")
        return None

if __name__ == '__main__':
    # This part is for testing the module independently
    # You would typically call process_resumes from your main Streamlit app
    st.title("Resume Processor Test")

    # Mock API Key for local testing - replace with st.secrets or env var in app
    mock_api_key = st.text_input("Enter Google API Key for testing", type="password")

    if mock_api_key:
        genai.configure(api_key=mock_api_key) # Configure genai for this test block
        st.info("Google GenAI configured for testing.")

        # Test processing from folder
        if st.button("Process Resumes from '/resumes' Folder (Test)"):
            # Ensure the 'resumes' directory exists for testing, or create dummy files for the test
            if not os.path.exists(RESUMES_DIR_NAME):
                os.makedirs(RESUMES_DIR_NAME, exist_ok=True)
                st.info(f"Created '{RESUMES_DIR_NAME}' directory for testing. Add some sample PDF/DOCX resumes there.")
            
            with st.spinner("Processing resumes from folder..."):
                vector_store_instance = process_resumes_from_folder(mock_api_key)
                if vector_store_instance:
                    st.session_state.vector_store = vector_store_instance
                    st.success("Resumes from folder processed (testing)!")
                else:
                    st.error("Failed to process resumes from folder (testing).")

        if st.button("Load Existing Vector Store (Test)"):
            vector_store_instance = get_existing_vector_store(mock_api_key)
            if vector_store_instance:
                st.session_state.vector_store = vector_store_instance
                st.success("Vector store loaded for testing!")
            else:
                st.info("No vector store to load or error during loading.")

        if "vector_store" in st.session_state and st.session_state.vector_store:
            query = st.text_input("Test similarity search on loaded vector store:")
            if query:
                try:
                    docs = st.session_state.vector_store.similarity_search(query, k=2)
                    st.write("Similarity search results:")
                    for doc in docs:
                        st.write(f"Source: {doc.metadata.get('source', 'N/A')}")
                        st.write(doc.page_content[:200] + "...") # Display a snippet
                        st.markdown("---")
                except Exception as e:
                    st.error(f"Error during similarity search: {e}")
        else:
            st.info("No vector store in session. Process resumes or load an existing one to test search.")
    else:
        st.warning("Please enter a Google API Key to test resume processing features.") 