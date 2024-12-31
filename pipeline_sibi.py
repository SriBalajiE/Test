from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
from pathlib import Path
from qdrant_client import QdrantClient
from supabase import create_client, Client
import os
import tempfile
import logging

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Initialize models
llm = MistralAI(model="mistral-small-latest", api_key=MISTRAL_API_KEY)
embed_model = FastEmbedEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Configure global settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.show_progress = False
Settings.num_output = 5

# Initialize chat memory
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# Crawling Functionality
def crawl_page(url, page_content, max_depth, current_depth, visited, user_id, agent_id):
    """Crawl a single page and extract its content"""
    try:
        # Parse HTML content
        soup = BeautifulSoup(page_content, 'html.parser')
        
        # Extract text content (excluding scripts, styles, etc.)
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()
        
        # Get clean text
        text = soup.get_text(separator='\n', strip=True)
        
        # Format content with URL and text
        page_content = f"\nURL: {url}\n\nContent:\n{text}\n"
        
        return page_content, get_internal_links(soup, url)
        
    except Exception as e:
        logging.error(f"Error crawling page {url}: {str(e)}")
        return "", []

def crawl(url, max_depth=3, current_depth=0, visited=None, user_id="1", agent_id="123"):
    """Crawl website and store content"""
    if visited is None:
        visited = set()
        
    try:
        domain = get_domain_from_url(url)
        file_name = f"{domain}_content.txt"
        storage_path = f"{user_id}/{agent_id}"
        local_file_path = f"{domain}_content.txt"
        
        # Create/clear local file
        with open(local_file_path, 'w', encoding='utf-8') as f:
            f.write("")  # Initialize empty file
            
        def process_url(url, depth):
            if depth > max_depth or url in visited:
                return
            
            visited.add(url)
            logging.info(f"Crawling URL: {url}")
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                content, internal_links = crawl_page(url, response.text, max_depth, depth, visited, user_id, agent_id)
                
                # Append content to local file
                with open(local_file_path, 'a', encoding='utf-8') as f:
                    f.write(content)
                
                # Recursively crawl internal links
                for link in internal_links:
                    if link not in visited and is_internal_link(url, link):
                        process_url(link, depth + 1)
                        
            except Exception as e:
                logging.error(f"Error crawling {url}: {str(e)}")
        
        # Start crawling from the initial URL
        process_url(url, current_depth)
        
        # After all crawling is complete, upload file to Supabase
        try:
            with open(local_file_path, 'rb') as f:
                file_data = f.read()
                # Delete existing file if it exists
                try:
                    supabase.storage.from_('Files').remove([f"{storage_path}/{file_name}"])
                    logging.info(f"Existing file deleted: {storage_path}/{file_name}")
                except:
                    pass
                
                # Upload new file
                supabase.storage.from_('Files').upload(
                    path=f"{storage_path}/{file_name}",
                    file=file_data
                )
                logging.info(f"Successfully uploaded file to: {storage_path}/{file_name}")
        except Exception as e:
            logging.error(f"Error uploading to Supabase: {str(e)}")
        finally:
            # Clean up local file
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
                
    except Exception as e:
        logging.error(f"Error in crawl function: {str(e)}")
        raise

    return visited

def get_page_identifier(url):
    """Get a unique identifier for a page, including its path"""
    parsed = urlparse(url)
    # Remove leading/trailing slashes and replace remaining slashes with underscores
    path = parsed.path.strip('/').replace('/', '_')
    # If it's the homepage (empty path), use 'home'
    if not path:
        path = 'home'
    return f"{parsed.netloc}_{path}"

def get_domain_from_url(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.split('.')[0]
    if domain.startswith('www'):
        domain = '.'.join(parsed_url.netloc.split('.')[1:])
    return domain

def is_internal_link(base_url, link_url):
    base_domain = urlparse(base_url).netloc
    link_domain = urlparse(link_url).netloc
    return base_domain == link_domain

def get_internal_links(soup, base_url):
    internal_links = []
    for anchor in soup.find_all('a', href=True):
        link = anchor['href']
        full_link = urljoin(base_url, link)
        if is_internal_link(base_url, full_link):
            internal_links.append(full_link)
    return internal_links

def create_or_load_index(collection_name: str, documents=None):
    """Create a new index or load existing one from Qdrant Cloud"""
    try:
        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name
        )

        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if documents:
            logging.info("Creating new index with documents")
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            logging.info("Index created and stored in Qdrant Cloud successfully")
        else:
            try:
                logging.info("Attempting to load existing index from Qdrant Cloud")
                index = load_index_from_storage(storage_context)  # Load from Qdrant
                logging.info("Successfully loaded index from Qdrant Cloud")
            except Exception as e:
                logging.info(f"Could not load existing index, creating empty one: {str(e)}")
                index = VectorStoreIndex.from_vector_store(
                    vector_store,
                    storage_context=storage_context,
                    show_progress=True
                )
                logging.info("Empty index created in Qdrant Cloud successfully")

        return index
            
    except Exception as e:
        logging.error(f"Error in create_or_load_index: {str(e)}")
        raise

def process_documents(files, storage_path):
    """Process and load documents from Supabase"""
    try:
        # Create temporary directory for document processing
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []
            for file in files:
                try:
                    # Download file from Supabase
                    file_data = supabase.storage.from_("Files").download(f"{storage_path}/{file['name']}")
                    if not file_data:
                        logging.warning(f"Could not download file: {file['name']}")
                        continue
                        
                    # Save to temporary directory
                    temp_file_path = Path(temp_dir) / file['name']
                    with open(temp_file_path, 'wb') as f:
                        f.write(file_data)
                    file_paths.append(temp_file_path)
                    logging.info(f"Successfully processed file: {file['name']}")
                except Exception as e:
                    logging.error(f"Error processing file {file['name']}: {str(e)}")
                    continue
            
            if not file_paths:
                raise Exception("No files were successfully processed")
                
            # Load documents using SimpleDirectoryReader
            documents = SimpleDirectoryReader(
                input_files=file_paths
            ).load_data()
            
            return documents
    except Exception as e:
        logging.error(f"Error processing documents: {str(e)}")
        raise

def create_index_for_customer(user_id: str, agent_id: str):
    """Create or update index for a customer"""
    try:
        collection_name = f"{user_id}_{agent_id}"
        storage_path = f"{user_id}/{agent_id}"
        
        # Fetch files from Supabase storage
        try:
            files = supabase.storage.from_("Files").list(storage_path)
            logging.info(f"Found {len(files)} files in storage path: {storage_path}")
        except Exception as e:
            logging.error(f"Error listing files from Supabase: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error accessing Supabase storage: {str(e)}"
            )
        
        if not files:
            raise HTTPException(
                status_code=404,
                detail=f"No files found in storage path: {storage_path}"
            )
            
        # Process the documents
        try:
            documents = process_documents(files, storage_path)
            logging.info(f"Successfully processed {len(documents)} documents")
        except Exception as e:
            logging.error(f"Error processing documents: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing documents: {str(e)}"
            )
        
        # Create or update the index
        create_or_load_index(collection_name, documents)
        logging.info(f"Successfully created/updated index for collection: {collection_name}")
        
        return {
            "status": "success",
            "message": "Index created successfully",
            "collection_id": collection_name,
            "document_count": len(documents)
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error in create_index_for_customer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def chat_response(question: str, collection_id: str, company_name: str = None):
    """Generate a streaming response for a chat message"""
    try:
        # Load the index
        index = create_or_load_index(collection_id)
        
        # Ensure memory is defined before using it
        if 'memory' not in locals():
            logging.error("Memory is not initialized.")
            raise RuntimeError("Memory is not initialized.")
        
        # Create chat engine with memory and system prompt
        chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=memory,
            system_prompt=(
                f"""You are an AI assistant employed by the company '{company_name}'. 
                1. Only provide information that is explicitly stated in the available information.
                2. If the customer is leaving the conversation, end the chat gracefully.
                3. If the customer is asking a question that is not related to the available information, you must reply exactly with 'Sorry, I can only provide information about {company_name}'.
                4. At the end of each response, always ask the user if they need further assistance about {company_name}.
                5. If the customer is asking a question that is not related to the available information, you must reply exactly with 'Sorry, I can only provide information about {company_name}'.
                Here is the available information:
                """
            ),
            streaming=True,
            verbose=True
        )
        
        # Generate streaming response
        response_stream = chat_engine.stream_chat(question)
        
        # Stream the response
        for response in response_stream.response_gen:
            # Send each chunk with SSE format
            yield f"data: {response}\n\n"
            
        # Update chat memory after streaming is complete
        memory.put(question, str(response_stream.response))
        
    except Exception as e:
        logging.error(f"Error generating chat response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI models
class QueryRequest(BaseModel):
    question: str
    collection_id: str = None
    company_name: str = None

class QueryResponse(BaseModel):
    answer: str

class RetrainRequest(BaseModel):
    user_id: str
    agent_id: str
    crawl_url: str = None
    company_name: str = None

# FastAPI endpoints
app = FastAPI()

@app.post("/query")
async def handle_query(request: QueryRequest):
    """Handle incoming queries with streaming response"""
    try:
        # Create generator for streaming response
        return StreamingResponse(
            chat_response(request.question, request.collection_id, request.company_name),
            media_type='text/event-stream'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain(request: RetrainRequest):
    """Retrain the model and optionally crawl a website"""
    try:
        if request.crawl_url:
            print(f"Crawling website: {request.crawl_url}")
            crawl(request.crawl_url, user_id=request.user_id, agent_id=request.agent_id)
            
        index = create_index_for_customer(request.user_id, request.agent_id)
        return {"status": "success", "message": f"Model retrained successfully for {request.company_name}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def get_initial_message():
    """Get initial message"""
    return {
        "message": "Welcome to the AI Assistant API",
        "endpoints": {
            "/query": "POST - Send a question to get AI assistance",
            "/retrain": "POST - Retrain the model with new data"
        }
    }
