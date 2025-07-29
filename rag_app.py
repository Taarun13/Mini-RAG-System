import streamlit as st
import fitz  # PyMuPDF
import os
import numpy as np
import faiss
import uuid
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from scipy.spatial.distance import jaccard, hamming
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
import pickle
import tempfile
import requests
import time
import json

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# === CONFIGURATION ===
EXTRACTED_TEXT_FILE = "extracted_text.txt"
PROCESSED_TEXT_FILE = "processed_text.txt"
FAISS_INDEX_FILE = "faiss_index.bin"
DOCUMENTS_FILE = "documents.pkl"
METADATA_FILE = "metadata.pkl"

# API Configuration
EMBEDDING_API_URL = "https://embeddings-service.up.railway.app"
EMBEDDING_ENDPOINT = f"{EMBEDDING_API_URL}/embed-text"
HEALTH_ENDPOINT = f"{EMBEDDING_API_URL}/health"

# API-based embedding functions
def check_embedding_service_health():
    """Check if the external embedding service is available"""
    try:
        st.info("🔗 Checking embedding service health...")
        response = requests.get(HEALTH_ENDPOINT, timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            st.success(f"✅ Embedding Service Status: {health_data.get('status', 'unknown')}")
            st.info(f"📊 Model: {health_data.get('model', 'unknown')}")
            st.info(f"🕐 Uptime: {health_data.get('uptime_seconds', 0):.1f} seconds")
            return True, health_data
        else:
            st.error(f"❌ Embedding service unhealthy - Status: {response.status_code}")
            return False, None
            
    except requests.exceptions.Timeout:
        st.error("❌ Embedding service timeout - Service may be slow or unavailable")
        return False, None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to embedding service - Check internet connection")
        return False, None
    except Exception as e:
        st.error(f"❌ Error checking embedding service: {str(e)}")
        return False, None

def get_embedding_from_api(text, max_retries=3, retry_delay=1):
    """Get embedding using the external API service"""
    try:
        # Clean and validate text
        if not text or len(text.strip()) < 3:
            return None
        
        # Limit text length to avoid API issues
        text = text[:8000] if len(text) > 8000 else text
        
        # Prepare request payload
        payload = {
            "text": text.strip()
        }
        
        # Make API request with retries
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    EMBEDDING_ENDPOINT,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get('embedding', [])
                    
                    if embedding and len(embedding) > 0:
                        return np.array(embedding)
                    else:
                        st.error("❌ Empty embedding received from API")
                        return None
                        
                elif response.status_code == 400:
                    error_data = response.json()
                    st.error(f"❌ API Error: {error_data.get('message', 'Invalid input')}")
                    return None
                    
                elif response.status_code == 422:
                    error_data = response.json()
                    st.error(f"❌ Validation Error: {error_data}")
                    return None
                    
                elif response.status_code == 503:
                    st.warning(f"⚠️ Service unavailable, retrying... (Attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        st.error("❌ Service unavailable after all retries")
                        return None
                        
                else:
                    st.error(f"❌ API Error: Status {response.status_code}")
                    return None
                    
            except requests.exceptions.Timeout:
                st.warning(f"⚠️ Request timeout, retrying... (Attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    st.error("❌ Request timeout after all retries")
                    return None
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Connection error - Check internet connection")
                return None
                
        return None
            
    except Exception as e:
        st.error(f"❌ Embedding API error: {str(e)}")
        return None

# Updated cache function for API embeddings
@st.cache_data
def get_cached_embedding(text):
    """Cache embeddings to avoid repeated API calls"""
    return get_embedding_from_api(text)

def get_embedding_from_model(text, model=None):
    """Updated function to use API instead of local model"""
    return get_embedding_from_api(text)

# Initialize embedding service (replaces local model loading)
@st.cache_resource
def initialize_embedding_service():
    """Initialize and check the embedding API service"""
    try:
        st.info("🔄 Initializing embedding service connection...")
        
        is_healthy, health_data = check_embedding_service_health()
        
        if is_healthy:
            st.success("✅ Embedding service initialized successfully!")
            return True, health_data
        else:
            st.error("❌ Failed to initialize embedding service")
            return False, None
            
    except Exception as e:
        st.error(f"❌ Error initializing embedding service: {str(e)}")
        return False, None

# === STEP 1: PDF LOADING AND TEXT EXTRACTION ===
def extract_text_from_pdf(pdf_file):
    """Extract all contents from PDF and save to text file"""
    doc = None
    try:
        st.info("📖 Step 1: Loading PDF and extracting text...")
        
        # Create a temporary file to work with the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Open PDF document
            doc = fitz.open(tmp_path)
            total_pages = len(doc)
            full_text = ""
            
            # Progress bar for extraction
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Extract text from each page
            for page_num in range(total_pages):
                status_text.text(f"Processing page {page_num + 1} of {total_pages}")
                
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    if page_text.strip():  # Only add non-empty pages
                        full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    
                    progress_bar.progress((page_num + 1) / total_pages)
                    
                except Exception as page_error:
                    st.warning(f"⚠️ Error processing page {page_num + 1}: {str(page_error)}")
                    continue
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Check if we extracted any text
            if not full_text.strip():
                st.error("❌ No text found in PDF")
                return None, 0
            
            # Save extracted text to file
            try:
                with open(EXTRACTED_TEXT_FILE, "w", encoding="utf-8") as f:
                    f.write(full_text)
                st.success(f"📄 Text saved to: {EXTRACTED_TEXT_FILE}")
            except Exception as save_error:
                st.error(f"❌ Error saving text file: {str(save_error)}")
            
            st.success(f"✅ Step 1 Complete: Extracted text from {total_pages} pages")
            
            return full_text, total_pages
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    except Exception as e:
        st.error(f"❌ Error in Step 1: {str(e)}")
        return None, 0
    
    finally:
        # Always close the document if it was opened
        if doc is not None:
            try:
                doc.close()
            except:
                pass

# === STEP 2: TEXT PREPROCESSING AND FAISS VECTOR STORAGE ===
def preprocess_text(text):
    """Lemmatize and stem the text"""
    try:
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        
        # Tokenize and process
        sentences = sent_tokenize(text)
        processed_sentences = []
        
        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            processed_tokens = [
                stemmer.stem(lemmatizer.lemmatize(word))
                for word in tokens
                if word.isalnum() and word not in stop_words and len(word) > 2
            ]
            if processed_tokens:  # Only add non-empty sentences
                processed_sentences.append(" ".join(processed_tokens))
        
        processed_text = "\n".join(processed_sentences)
        
        # Save processed text
        with open(PROCESSED_TEXT_FILE, "w", encoding="utf-8") as f:
            f.write(processed_text)
        
        return processed_text, processed_sentences
    
    except Exception as e:
        st.error(f"❌ Error in text preprocessing: {str(e)}")
        return None, []

def chunk_text(text, chunk_size=500, overlap=100):
    """Create overlapping chunks from text"""
    words = text.split()
    chunks = []
    
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        if end >= len(words):
            break
        start += chunk_size - overlap
    
    return chunks

def store_in_faiss_db(processed_text):
    """Convert to embeddings using API and store in FAISS"""
    try:
        st.info("🔄 Step 2: Processing text and storing in FAISS vector database...")
        
        # Check embedding service
        service_ready, health_data = initialize_embedding_service()
        if not service_ready:
            st.error("❌ Cannot initialize embedding service")
            return False, 0
        
        # Create chunks
        chunks = chunk_text(processed_text)
        st.info(f"📄 Created {len(chunks)} text chunks")
        
        if not chunks:
            st.error("❌ No chunks created from text")
            return False, 0
        
        # Test API connectivity and get embedding dimension
        st.info("🔗 Testing embedding API...")
        test_embedding = get_embedding_from_api("test connection")
        if test_embedding is None:
            st.error("❌ Cannot generate test embedding from API")
            return False, 0
        else:
            embedding_dim = len(test_embedding)
            st.success(f"✅ API working! Embedding dimension: {embedding_dim}")
        
        # Initialize FAISS index
        try:
            # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            # You can also use IndexFlatL2 for L2 distance
            index = faiss.IndexFlatIP(embedding_dim)
            st.success("✅ FAISS index initialized")
            
        except Exception as faiss_error:
            st.error(f"❌ FAISS initialization error: {str(faiss_error)}")
            return False, 0
        
        # Process chunks with API calls
        embeddings = []
        valid_chunks = []
        chunk_metadata = []
        failed_chunks = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Add rate limiting info
        st.info("⏱️ Processing chunks with API calls (this may take longer than local processing)...")
        
        for i, chunk in enumerate(chunks):
            status_text.text(f"API call {i + 1}/{len(chunks)} - Processing chunk...")
            
            # Skip very short chunks
            if len(chunk.strip()) < 10:
                failed_chunks += 1
                continue
            
            # Get embedding from API
            embedding = get_embedding_from_api(chunk)
            
            if embedding is not None:
                # Normalize embedding for cosine similarity with IndexFlatIP
                embedding_normalized = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding_normalized)
                valid_chunks.append(chunk)
                chunk_metadata.append({
                    'id': f"chunk_{i}_{str(uuid.uuid4())[:8]}",
                    'chunk_index': i,
                    'text_length': len(chunk)
                })
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.1)
            else:
                failed_chunks += 1
                st.warning(f"⚠️ Failed to get embedding for chunk {i + 1}")
            
            progress_bar.progress((i + 1) / len(chunks))
            
            # Show progress every 5 chunks
            if len(valid_chunks) > 0 and len(valid_chunks) % 5 == 0:
                st.info(f"📊 Processed {len(valid_chunks)} chunks so far...")
        
        progress_bar.empty()
        status_text.empty()
        
        # Store in FAISS if we have valid embeddings
        if valid_chunks and embeddings:
            try:
                # Convert embeddings to numpy array
                embeddings_array = np.array(embeddings).astype('float32')
                
                # Add to FAISS index
                index.add(embeddings_array)
                
                # Save FAISS index
                faiss.write_index(index, FAISS_INDEX_FILE)
                
                # Save documents and metadata separately
                with open(DOCUMENTS_FILE, 'wb') as f:
                    pickle.dump(valid_chunks, f)
                
                with open(METADATA_FILE, 'wb') as f:
                    pickle.dump(chunk_metadata, f)
                
                st.success(f"✅ Step 2 Complete!")
                st.success(f"📊 Successfully stored {len(valid_chunks)} embeddings in FAISS")
                
                if failed_chunks > 0:
                    st.warning(f"⚠️ {failed_chunks} chunks failed to process")
                
                # Show storage summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Chunks", len(chunks))
                with col2:
                    st.metric("Successful", len(valid_chunks))
                with col3:
                    st.metric("Failed", failed_chunks)
                
                # Show FAISS index info
                st.info(f"📊 FAISS Index Info:")
                st.info(f"- Dimension: {index.d}")
                st.info(f"- Total vectors: {index.ntotal}")
                st.info(f"- Index type: {type(index).__name__}")
                
                return True, len(valid_chunks)
                
            except Exception as store_error:
                st.error(f"❌ Error storing in FAISS: {str(store_error)}")
                return False, 0
        else:
            st.error("❌ No valid embeddings created")
            st.error("Possible issues:")
            st.error("1. All chunks too short or empty")
            st.error("2. API service problems")
            st.error("3. Internet connectivity issues")
            st.error("4. Text preprocessing removed all content")
            
            # Debug info
            if chunks:
                st.info(f"📊 Debug info:")
                st.info(f"- Total chunks: {len(chunks)}")
                st.info(f"- Failed chunks: {failed_chunks}")
                st.info(f"- First chunk preview: {chunks[0][:100]}...")
            
            return False, 0
    
    except Exception as e:
        st.error(f"❌ Error in Step 2: {str(e)}")
        return False, 0

# === STEP 3: USER QUERY PROCESSING ===
def process_user_query(query):
    """Get user query and convert to embeddings using API"""
    try:
        st.info("🔍 Step 3: Processing user query...")
        
        # Check embedding service
        service_ready, health_data = initialize_embedding_service()
        if not service_ready:
            st.error("❌ Cannot connect to embedding service")
            return None, query
        
        # Preprocess query
        processed_query, _ = preprocess_text(query)
        if not processed_query:
            processed_query = query.lower()
        
        # Get query embedding from API
        query_embedding = get_embedding_from_api(processed_query)
        
        if query_embedding is not None:
            # Normalize for cosine similarity
            query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
            st.success("✅ Step 3 Complete: Query converted to embedding via API")
            return query_embedding_normalized, processed_query
        else:
            st.error("❌ Failed to get query embedding from API")
            return None, processed_query
    
    except Exception as e:
        st.error(f"❌ Error in Step 3: {str(e)}")
        return None, query

# === STEP 4 & 5: SIMILARITY COMPARISON WITH FAISS ===
def compute_all_similarities(query_embedding, doc_embeddings):
    """Compute multiple similarity metrics"""
    try:
        # Convert to numpy arrays and ensure proper shapes
        query_emb = np.array(query_embedding).reshape(1, -1)
        doc_embs = np.array(doc_embeddings)
        
        # Ensure doc_embs is 2D
        if len(doc_embs.shape) == 1:
            doc_embs = doc_embs.reshape(1, -1)
        
        # 1. Cosine Similarity
        cosine_scores = cosine_similarity(query_emb, doc_embs).flatten()
        
        # 2. Euclidean Distance (converted to similarity)
        euclidean_distances_scores = euclidean_distances(query_emb, doc_embs).flatten()
        euclidean_similarities = 1 / (1 + euclidean_distances_scores)  # Convert to similarity
        
        # 3. Manhattan Distance (converted to similarity)  
        manhattan_distances_scores = manhattan_distances(query_emb, doc_embs).flatten()
        manhattan_similarities = 1 / (1 + manhattan_distances_scores)  # Convert to similarity
        
        # 4. Dot Product Similarity
        query_flat = query_embedding.flatten() if len(query_embedding.shape) > 1 else query_embedding
        dot_product_scores = np.array([np.dot(doc_emb, query_flat) for doc_emb in doc_embs])
        
        # 5. Pearson Correlation
        try:
            pearson_scores = []
            query_flat = query_embedding.flatten() if len(query_embedding.shape) > 1 else query_embedding
            for doc_emb in doc_embs:
                doc_flat = doc_emb.flatten() if len(doc_emb.shape) > 1 else doc_emb
                corr_matrix = np.corrcoef(query_flat, doc_flat)
                corr = corr_matrix[0, 1] if corr_matrix.size > 1 else 0
                pearson_scores.append(corr if not np.isnan(corr) else 0)
            pearson_scores = np.array(pearson_scores)
        except:
            pearson_scores = np.zeros(len(doc_embs))
        
        return {
            'cosine': cosine_scores,
            'euclidean': euclidean_similarities,
            'manhattan': manhattan_similarities,
            'dot_product': dot_product_scores,
            'pearson': pearson_scores
        }
    
    except Exception as e:
        st.error(f"❌ Error computing similarities: {str(e)}")
        return None

def search_similar_documents_faiss(query_embedding, top_k=5):
    """Search for similar documents using FAISS"""
    try:
        st.info("🔍 Step 4-5: Searching with FAISS...")
        
        # Check if FAISS index files exist
        if not os.path.exists(FAISS_INDEX_FILE):
            st.error("❌ FAISS index file not found. Please process a PDF first.")
            return None
        
        if not os.path.exists(DOCUMENTS_FILE):
            st.error("❌ Documents file not found. Please process a PDF first.")
            return None
        
        if not os.path.exists(METADATA_FILE):
            st.error("❌ Metadata file not found. Please process a PDF first.")
            return None
        
        # Load FAISS index
        try:
            index = faiss.read_index(FAISS_INDEX_FILE)
            st.info(f"📊 Loaded FAISS index with {index.ntotal} vectors")
        except Exception as faiss_error:
            st.error(f"❌ Error loading FAISS index: {str(faiss_error)}")
            return None
        
        # Load documents and metadata
        try:
            with open(DOCUMENTS_FILE, 'rb') as f:
                documents = pickle.load(f)
            
            with open(METADATA_FILE, 'rb') as f:
                metadata = pickle.load(f)
                
            st.info(f"📊 Loaded {len(documents)} documents")
        except Exception as load_error:
            st.error(f"❌ Error loading documents/metadata: {str(load_error)}")
            return None
        
        # Ensure we have embeddings to compare against
        if index.ntotal == 0:
            st.error("❌ FAISS index is empty")
            return None
        
        if len(documents) != index.ntotal:
            st.error(f"❌ Mismatch: {len(documents)} documents vs {index.ntotal} vectors in index")
            return None
        
        # Prepare query embedding for FAISS search
        query_emb = np.array(query_embedding).astype('float32').reshape(1, -1)
        
        # Perform FAISS search
        try:
            # Search with FAISS (this gives us cosine similarity scores since we used IndexFlatIP)
            faiss_scores, faiss_indices = index.search(query_emb, min(top_k, index.ntotal))
            
            # Flatten the results
            faiss_scores = faiss_scores.flatten()
            faiss_indices = faiss_indices.flatten()
            
            st.info(f"📊 FAISS search completed")
            st.info(f"📊 Query embedding shape: {query_emb.shape}")
            
        except Exception as search_error:
            st.error(f"❌ FAISS search error: {str(search_error)}")
            return None
        
        # Get all embeddings for additional similarity metrics
        try:
            # Reconstruct embeddings from index (for IndexFlat types)
            all_embeddings = np.zeros((index.ntotal, index.d), dtype='float32')
            index.reconstruct_n(0, index.ntotal, all_embeddings)
            
        except Exception as reconstruct_error:
            st.warning(f"⚠️ Could not reconstruct embeddings for additional metrics: {str(reconstruct_error)}")
            # Fallback: only use FAISS results
            results = {
                'faiss_cosine': {
                    'indices': faiss_indices,
                    'scores': faiss_scores,
                    'documents': [documents[i] for i in faiss_indices if i < len(documents)]
                }
            }
            st.success("✅ Step 4-5 Complete: FAISS similarity search done")
            return results
        
        # Compute all similarity metrics using reconstructed embeddings
        similarities = compute_all_similarities(query_embedding, all_embeddings)
        
        if similarities is None:
            # Fallback to FAISS results only
            results = {
                'faiss_cosine': {
                    'indices': faiss_indices,
                    'scores': faiss_scores,
                    'documents': [documents[i] for i in faiss_indices if i < len(documents)]
                }
            }
        else:
            # Get top results for each metric
            results = {}
            
            # Add FAISS results first
            results['faiss_cosine'] = {
                'indices': faiss_indices,
                'scores': faiss_scores,
                'documents': [documents[i] for i in faiss_indices if i < len(documents)]
            }
            
            # Add other similarity metrics
            for metric_name, scores in similarities.items():
                # Ensure scores is a numpy array
                scores_array = np.array(scores)
                
                # Get top indices
                top_indices = np.argsort(-scores_array)[:top_k]
                
                # Build results
                results[metric_name] = {
                    'indices': top_indices,
                    'scores': scores_array[top_indices],
                    'documents': [documents[i] for i in top_indices if i < len(documents)]
                }
        
        st.success("✅ Step 4-5 Complete: FAISS similarity search and comparison done")
        return results
    
    except Exception as e:
        st.error(f"❌ Error in FAISS similarity search: {str(e)}")
        import traceback
        st.error(f"❌ Detailed error: {traceback.format_exc()}")
        return None

# === STEP 6-7: DISPLAY RESULTS ===
def display_similarity_results(results, original_query):
    """Display results of all similarity measurements"""
    if not results:
        st.error("❌ No results to display")
        return
    
    st.subheader("📊 Step 6-7: Similarity Measurement Results")
    st.markdown(f"**Original Query:** {original_query}")
    
    # Create tabs for different similarity metrics
    metric_names = list(results.keys())
    tabs = st.tabs([f"📊 {name.replace('_', ' ').title()}" for name in metric_names])
    
    for tab, metric_name in zip(tabs, metric_names):
        with tab:
            st.markdown(f"### {metric_name.replace('_', ' ').title()} Results")
            
            # Special handling for FAISS results
            if metric_name == 'faiss_cosine':
                st.info("🔍 These are the primary FAISS search results (optimized for speed)")
            
            metric_results = results[metric_name]
            
            for i, (score, document) in enumerate(zip(metric_results['scores'], metric_results['documents'])):
                with st.expander(f"Rank #{i+1} - Score: {score:.4f}", expanded=(i==0)):
                    st.markdown(f"**Similarity Score:** `{score:.4f}`")
                    st.markdown(f"**Document Index:** {metric_results['indices'][i]}")
                    
                    # Additional info for FAISS results
                    if metric_name == 'faiss_cosine':
                        st.markdown("**Source:** FAISS Vector Search")
                    
                    st.markdown("**Content Preview:**")
                    
                    # Show first 1000 characters
                    preview = document[:1000] + "..." if len(document) > 1000 else document
                    st.markdown(f"```\n{preview}\n```")
                    
                    # Option to show full content
                    if st.button(f"Show Full Content - {metric_name} #{i+1}", key=f"{metric_name}_{i}"):
                        st.markdown("**Full Content:**")
                        st.text_area("", document, height=200, key=f"full_{metric_name}_{i}")
    
    # Summary comparison
    st.subheader("📈 Similarity Metrics Summary")
    
    summary_data = {}
    for metric_name, metric_results in results.items():
        summary_data[metric_name] = {
            'Max Score': f"{np.max(metric_results['scores']):.4f}",
            'Min Score': f"{np.min(metric_results['scores']):.4f}",
            'Avg Score': f"{np.mean(metric_results['scores']):.4f}"
        }
    
    import pandas as pd
    summary_df = pd.DataFrame(summary_data).T
    st.dataframe(summary_df)
    
    # FAISS Performance Note
    if 'faiss_cosine' in results:
        st.info("💡 **FAISS Performance Note:** The 'Faiss Cosine' results are optimized for fast similarity search and should be your primary reference for document retrieval.")

# === MAIN STREAMLIT APP ===
def main():
    st.set_page_config(page_title="Complete RAG Workflow - FAISS", layout="wide")
    
    st.title("🤖 Complete RAG Workflow System")
    st.markdown("*PDF Processing → FAISS Vector Storage → Query Processing → Similarity Search*")
    st.markdown("**Using Google Gemini Embeddings API + FAISS Vector Database**")
    
    # Sidebar configuration and file upload
    st.sidebar.header("📁 PDF File Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF file to process and analyze"
    )
    
    # Display file information if uploaded
    if uploaded_file is not None:
        st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
        st.sidebar.info(f"📊 File size: {uploaded_file.size:,} bytes")
    else:
        st.sidebar.info("👆 Please upload a PDF file to get started")
    
    st.sidebar.header("⚙️ Configuration")
    chunk_size = st.sidebar.slider("Chunk Size", 200, 1000, 500)
    top_k = st.sidebar.slider("Top K Results", 1, 10, 5)
    
    # FAISS Index Type Selection
    index_type = st.sidebar.selectbox(
        "FAISS Index Type",
        ["IndexFlatIP", "IndexFlatL2", "IndexIVFFlat"],
        help="IndexFlatIP: Inner Product (Cosine), IndexFlatL2: L2 Distance, IndexIVFFlat: Faster approximate search"
    )
    
    # API Status in sidebar
    st.sidebar.header("🌐 API Status")
    if st.sidebar.button("🔄 Check API Health"):
        is_healthy, health_data = check_embedding_service_health()
        if is_healthy:
            st.sidebar.success("✅ API Service Online")
        else:
            st.sidebar.error("❌ API Service Offline")
    
    # Initialize session state
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'faiss_ready' not in st.session_state:
        st.session_state.faiss_ready = False
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    
    # Check if a new file was uploaded
    if uploaded_file is not None:
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.pdf_processed = False
            st.session_state.faiss_ready = False
            st.info(f"🔄 New file detected: {uploaded_file.name}")
    
    # Main content area
    if uploaded_file is None:
        st.info("👈 Please upload a PDF file from the sidebar to begin processing")
        st.markdown("""
        ### How to use this application:
        1. **Upload PDF**: Use the sidebar to upload your PDF file
        2. **Process PDF**: Click the 'Process PDF' button to extract and vectorize content using API
        3. **Ask Questions**: Enter your queries to search through the document
        4. **View Results**: Explore similarity results across different metrics
        
        ### ⚡ Using Google Gemini Embeddings API + FAISS
        - **Higher Quality**: Google's advanced embedding model
        - **Fast Search**: FAISS provides optimized vector similarity search
        - **Multiple Metrics**: Compare results across different similarity measures
        - **Cloud-powered**: No local model download required
        - **Internet Required**: Needs stable internet connection for embeddings
        - **Scalable**: FAISS can handle large document collections efficiently
        
        ### 🚀 FAISS Advantages:
        - **Speed**: Much faster similarity search than traditional methods
        - **Memory Efficient**: Optimized memory usage for large vector collections
        - **Scalable**: Can handle millions of vectors
        - **Multiple Index Types**: Choose the best index for your use case
        - **GPU Support**: Can utilize GPU acceleration (if available)
        
        """)
    else:
        # Step 1 & 2: Process PDF
        st.header("📄 PDF Processing")
        
        if st.button("🚀 Process PDF (Steps 1-2)", disabled=uploaded_file is None):
            if uploaded_file is not None:
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                
                # Step 1: Extract text
                extracted_text, num_pages = extract_text_from_pdf(uploaded_file)
                
                if extracted_text:
                    st.session_state.pdf_processed = True
                    
                    # Step 2: Preprocess and store in FAISS
                    processed_text, sentences = preprocess_text(extracted_text)
                    
                    if processed_text:
                        success, num_embeddings = store_in_faiss_db(processed_text)
                        
                        if success:
                            st.session_state.faiss_ready = True
                            
                            # Show summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("PDF Pages", num_pages)
                            with col2:
                                st.metric("Text Sentences", len(sentences))
                            with col3:
                                st.metric("FAISS Vectors", num_embeddings)
        
        # Steps 3-7: Query Processing
        if st.session_state.faiss_ready:
            st.header("💬 Query Processing (Steps 3-7)")
            
            user_query = st.text_input(
                "Enter your question:",
                placeholder="Ask anything about the PDF content..."
            )
            
            if user_query and st.button("🔍 Search Similar Content"):
                # Step 3: Process query
                query_embedding, processed_query = process_user_query(user_query)
                
                if query_embedding is not None:
                    # Steps 4-5: Search and compare using FAISS
                    search_results = search_similar_documents_faiss(query_embedding, top_k)
                    
                    if search_results:
                        # Steps 6-7: Display results
                        display_similarity_results(search_results, user_query)
        
        elif st.session_state.pdf_processed:
            st.info("⏳ PDF processed but FAISS database not ready. Please try processing again.")
        else:
            st.info("👆 Please process the uploaded PDF file first to enable querying.")
    
    # Status sidebar
    st.sidebar.header("📊 System Status")
    st.sidebar.success("✅ PDF Processed" if st.session_state.pdf_processed else "⏳ PDF Not Processed")
    st.sidebar.success("✅ FAISS Ready" if st.session_state.faiss_ready else "⏳ FAISS Not Ready")
    
    # File status
    if os.path.exists(EXTRACTED_TEXT_FILE):
        st.sidebar.text("📄 extracted_text.txt ✅")
    if os.path.exists(PROCESSED_TEXT_FILE):
        st.sidebar.text("📄 processed_text.txt ✅")
    if os.path.exists(FAISS_INDEX_FILE):
        st.sidebar.text("🗄️ FAISS Index ✅")
    if os.path.exists(DOCUMENTS_FILE):
        st.sidebar.text("📚 Documents ✅")
    if os.path.exists(METADATA_FILE):
        st.sidebar.text("📋 Metadata ✅")
    
    # FAISS Management
    st.sidebar.header("🗄️ FAISS Management")
    
    if st.sidebar.button("🗑️ Clear FAISS Database"):
        try:
            files_to_remove = [FAISS_INDEX_FILE, DOCUMENTS_FILE, METADATA_FILE]
            removed_files = []
            
            for file_path in files_to_remove:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    removed_files.append(file_path)
            
            if removed_files:
                st.sidebar.success(f"✅ Cleared {len(removed_files)} FAISS files")
                st.session_state.faiss_ready = False
            else:
                st.sidebar.info("ℹ️ No FAISS files to clear")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error clearing FAISS files: {str(e)}")
    
    # Display FAISS Index Info
    if os.path.exists(FAISS_INDEX_FILE):
        try:
            index = faiss.read_index(FAISS_INDEX_FILE)
            st.sidebar.header("📊 FAISS Index Info")
            st.sidebar.text(f"Vectors: {index.ntotal}")
            st.sidebar.text(f"Dimension: {index.d}")
            st.sidebar.text(f"Type: {type(index).__name__}")
            
            # Calculate approximate memory usage
            memory_mb = (index.ntotal * index.d * 4) / (1024 * 1024)  # 4 bytes per float32
            st.sidebar.text(f"Memory: ~{memory_mb:.1f} MB")
            
        except Exception as e:
            st.sidebar.error(f"Error reading FAISS info: {str(e)}")

if __name__ == "__main__":
    main()