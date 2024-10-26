import logging
import json
import os
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import numpy as np
from config import GEMINI_API_KEY
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import getpass
# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingProvider:
    def __init__(self, provider: str = 'gemini'):
        self.provider = provider.lower()
        if provider == 'gemini':
            self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            self.embedding_dim = 768  # Adjust based on model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_embedding(self, text: str) -> np.ndarray:
        '''Get embedding from the selected provider with retry logic'''
        try:
            embedding = self.embedding_model.embed_query(text)
            return np.array(embedding).reshape(1, -1)
        except Exception as e:
            logging.error(f"Error getting embedding: {e}")
            raise
    
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 5) -> np.ndarray:
        """Get embeddings for a batch of texts"""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = [self.get_embedding(text) for text in batch]
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

def create_faiss_index(texts: List[str], embeddings_dir: str) -> None:
    '''Create FAISS index from texts and save both index and raw texts'''
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create Document objects
    documents = [Document(page_content=text) for text in texts]
    
    # Create and save FAISS index
    vectorstore = FAISS.from_documents(documents, embedding_model)
    
    # Create directory if it doesn't exist
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Save the index and documents
    vectorstore.save_local(embeddings_dir)
    
    # Save raw texts
    with open(os.path.join(embeddings_dir, 'passages.json'), 'w') as f:
        json.dump(texts, f)

def load_and_split_data(file_path: str) -> List[str]:
    '''Load text data from a file and split into chunks'''
    with open(file_path, 'r') as file:
        text = file.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    return text_splitter.split_text(text)

def main(file_path: str, embeddings_dir: str):
    '''Main function to process text data and create FAISS index'''
    # Load and split the data
    texts = load_and_split_data(file_path)
    
    # Create and save the FAISS index
    create_faiss_index(texts, embeddings_dir)
    logging.info("FAISS index created successfully")

if __name__ == "__main__":
    main('context.txt', 'embeddings')