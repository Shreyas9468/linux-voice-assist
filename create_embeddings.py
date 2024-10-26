import logging
import faiss
import numpy as np
import json
import os
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import google.generativeai as genai
from config import GEMINI_API_KEY, GROQ_API_KEY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingProvider:
    def __init__(self, provider: str = 'gemini'):
        self.provider = provider.lower()
        if provider == 'gemini':
            genai.configure(api_key=GEMINI_API_KEY)
            self.embedding_dim = 768  # Dimension for text-embedding-004
        else:
            self.api_key = GROQ_API_KEY
            self.embedding_dim = 1024
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from the selected provider with retry logic"""
        if self.provider == 'gemini':
            try : 
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document",
                    title="Embedding generation"
                )
                embedding = np.array(result['embedding'])
                logging.info(f"Generated embedding for: {text}")
            except Exception as e:
                logging.error(f"Failed to generate embedding: {e}")
                raise

        else:  # groq
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': 'mixtral-8x7b',
                'input': text
            }
            response = requests.post(
                'https://api.groq.com/v1/embeddings',
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            embedding = np.array(data['data'][0]['embedding'])
        
        logging.debug(f"[ D E B U G ]Generated embedding for: {text}")
        logging.debug(f"[ D E B U G ]Embedding shape: {embedding.shape}")
        logging.debug(f"[ D E B U G ]Embedding: {embedding}")
        return embedding

    def get_batch_embeddings(self, texts: List[str], batch_size: int = 5) -> np.ndarray:
        """Get embeddings for a batch of texts"""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = [self.get_embedding(text) for text in batch]
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

def create_embeddings(context_file: str = 'context.txt'):
    """
    Create embeddings from context file using remote embedding service.
    Uses hardcoded values for simplicity.
    """
    # Configuration
    PROVIDER = 'gemini'  # or 'groq'
    OUTPUT_DIR = 'embeddings'
    CHUNK_SIZE = 512
    BATCH_SIZE = 5
    
    logging.info(f"Starting embeddings creation process using {PROVIDER}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize the embedding provider
    embedding_service = EmbeddingProvider(PROVIDER)
    
    # Read and chunk the context file
    logging.info(f"Reading context file: {context_file}")
    with open(context_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Text chunking with overlap
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1
        current_chunk.append(word)
        
        if current_length >= CHUNK_SIZE:
            chunks.append(' '.join(current_chunk))
            # Add 10% overlap with previous chunk
            overlap_size = len(current_chunk) // 10
            current_chunk = current_chunk[-overlap_size:]
            current_length = sum(len(word) + 1 for word in current_chunk)
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    logging.info(f"Created {len(chunks)} text chunks")
    
    # Generate embeddings in batches
    logging.info("Generating embeddings...")
    embeddings = embedding_service.get_batch_embeddings(chunks, BATCH_SIZE)
    
    # Create and save FAISS index
    logging.info("Creating FAISS index...")
    index = faiss.IndexFlatL2(embedding_service.embedding_dim)
    index.add(np.float32(embeddings))
    
    # Save index
    index_path = os.path.join(OUTPUT_DIR, 'faiss_index.bin')
    faiss.write_index(index, index_path)
    logging.info(f"Saved FAISS index to {index_path}")
    
    # Save passages and metadata
    passages_path = os.path.join(OUTPUT_DIR, 'passages.json')
    metadata = {
        'chunks': chunks,
        'provider': PROVIDER,
        'model': "models/text-embedding-004" if PROVIDER == 'gemini' else "mixtral-8x7b",
        'embedding_dim': embedding_service.embedding_dim,
        'num_chunks': len(chunks),
        'chunk_size': CHUNK_SIZE,
        'overlap': True,
        'overlap_size': '10%'
    }
    
    with open(passages_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved passages and metadata to {passages_path}")
    
    logging.info("Embedding creation completed successfully!")

if __name__ == "__main__":
    create_embeddings()