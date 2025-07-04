import logging
import chromadb
import tiktoken
import time
import os
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz
import traceback  # Added for better error logging

import nltk
nltk.download('punkt_tab', quiet=True)

from llm_inference import LLMInference
import db_utils

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('API_KEY')

class RAGSystem:
    def __init__(
            self, collection_name: str, 
            db_path: str ="PDF_ChromaDB", 
            n_results: int = 10  # Increased default results for summaries
        ):
        self.collection_name = collection_name
        self.db_path = db_path
        self.n_results = n_results

        if not self.collection_name:
            raise ValueError("'collection_name' parameter is required.")

        self.llm_inference = LLMInference()
        self.logger = self._setup_logging()
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.logger.info(f"Initialized ChromaDB collection: {self.collection_name} at {self.db_path}")
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        return logger
    
    def _format_time(self, response_time):
        minutes = response_time // 60
        seconds = response_time % 60
        return f"{int(minutes)}m {int(seconds)}s" if minutes else f"Time: {int(seconds)}s"
    
    def _generate_embeddings(self, text: str):
        try:
            self.logger.info(f"Generating embedding for text: {text[:50]}... with model: nomic-embed-text:latest")
            embedding = self.llm_inference._generate_embeddings(input_text=text, model_name="nomic-embed-text:latest")
            if not embedding:
                self.logger.error(f"Failed to generate embedding for text: {text[:50]}...")
                return []
            self.logger.debug(f"Generated embedding (first 5 values): {embedding[:5]}")
            return embedding
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _get_tokens_count(self, text: str):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    
    def _extract_file_name_from_query(self, query: str, stored_documents: list) -> str:
        query_lower = query.lower()
        query_lower = query_lower.replace("what is inside of the ", "").replace("what is inside ", "")
        query_lower = query_lower.replace("contents of ", "").replace("content of ", "")
        query_lower = query_lower.replace("summary of the ", "").replace("summary of ", "")  # Added for summary queries
        query_lower = query_lower.replace("document", "").replace("file", "").replace("the ", "").strip()
        
        file_name = None
        best_match_score = 0
        best_match_file = None

        stored_file_names = [doc[0] for doc in stored_documents]
        self.logger.info(f"Stored document names: {stored_file_names}")

        for stored_file in stored_file_names:
            stored_file_base = stored_file.lower().replace('.pdf', '').replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
            if stored_file_base in query_lower:
                self.logger.info(f"Exact match found for '{stored_file_base}' in query '{query}'")
                return stored_file
            score = fuzz.partial_ratio(query_lower, stored_file_base)
            self.logger.debug(f"Matching '{query_lower}' with '{stored_file_base}': score={score}")
            if score > best_match_score and score > 50:
                best_match_score = score
                best_match_file = stored_file

        file_name = best_match_file
        self.logger.info(f"Extracted file name from query '{query}': {file_name} (score: {best_match_score})")
        return file_name
    
    def _retrieve(self, user_text: str, stored_documents: list, n_results: int = 50):  # Increased to 50
        self.logger.info(f"Generating embeddings for query: {user_text}")
        embedding = self._generate_embeddings(user_text)
        if not embedding:
            self.logger.error("Failed to generate embeddings for query.")
            return [], []

        file_name = self._extract_file_name_from_query(user_text, stored_documents)
        where_filter = {"file_name": file_name} if file_name else None

        self.logger.info(f"Querying ChromaDB with embedding (first 5 values): {embedding[:5]} and filter: {where_filter}")
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "embeddings", "metadatas"],
            where=where_filter
        )

        if not results['documents']:
            self.logger.warning("No documents found in ChromaDB for the query with filter.")
            self.logger.info("Retrying query without file name filter...")
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                include=["documents", "embeddings", "metadatas"],
                where=None
            )

        if not results['documents']:
            self.logger.warning("No documents found in ChromaDB even without filter.")
            return [], []

        chunks = results['documents'][0]
        embeddings = results['embeddings'][0]
        self.logger.info(f"Retrieved {len(chunks)} document chunks: {chunks[:100]}...")
        return chunks, embeddings
    
    def _rerank_docs(self, chunks: list[str], embeddings: list[list[float]], query: str, top_k: int = 10):  # Increased to 10
        self.logger.info(f"Reranking {len(chunks)} chunks for query: {query}")
        tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores = bm25.get_scores(word_tokenize(query.lower()))

        query_embedding = np.array(self._generate_embeddings(query))
        chunk_embeddings = np.array(embeddings)

        dot_product = np.dot(chunk_embeddings, query_embedding)
        query_norm = np.linalg.norm(query_embedding)
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
        semantic_scores = dot_product / (chunk_norms * query_norm + 1e-10)

        bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-5)
        sem_norm = (semantic_scores - np.min(semantic_scores)) / (np.max(semantic_scores) - np.min(semantic_scores) + 1e-5)
        combined_scores = 0.5 * bm25_norm + 0.5 * sem_norm

        ranked_indices = np.argsort(combined_scores)[::-1]
        ranked_chunks = [chunks[i] for i in ranked_indices[:top_k]]
        self.logger.info(f"Top {top_k} reranked chunks: {ranked_chunks[:100]}...")
        return ranked_chunks
    
    def _get_prompt(self, query: str, context: str, is_summary: bool = False):
        if is_summary:
            prompt = f"""
You are an AI assistant specialized in summarizing documents based on the provided context. 
The context is a part of a document (PDF or image) and is structured with sections separated by `########`.

### **Context:**  
'''  
{context}  
'''  

### **Question:**  
"{query}"  

### **Instructions:**  
- Provide a detailed summary of the entire document based on the given context.  
- Include key sections such as introduction, methodology, results, and conclusion if present.  
- Do not omit any important details; ensure the summary is comprehensive.  

### **Answer:**

"""
        else:
            prompt = f"""
You are an AI assistant specialized in answering questions based **only** on the provided context. 
The context is a part of a document (PDF or image).
The context is structured with sections separated by `########`. 

### **Context:**  
'''  
{context}  
'''  

### **Question:**  
"{query}"  

### **Instructions:**  
- Answer concisely and accurately using only the given context.  
- Put what you find from the context **without summarizing**.
- Answer directly and concisely.

### **Answer:**

"""
        return prompt
    
    def generate_response(self, query: str, ollama_model):
        if not ollama_model:
            self.logger.error("No Ollama model provided.")
            return "Error: Choose an Ollama LLM", 0, 0, 0, 0

        self.logger.info(f"--> Generate Response Using Ollama LLM: {ollama_model}")
        stored_documents = db_utils.load_documents_from_db()
        if not stored_documents:
            self.logger.warning("No documents found in the database.")
            return "No documents available. Please upload a document first.", 0, 0, 0, 0

        chunks, embeddings = self._retrieve(query, stored_documents, n_results=50)
        
        if not chunks:
            self.logger.warning("No relevant chunks retrieved for the query.")
            return "No relevant information found. Please ensure documents are uploaded or rephrase your query.", 0, 0, 0, 0
        
        reranked_retrieved_docs = self._rerank_docs(chunks=chunks, embeddings=embeddings, query=query, top_k=self.n_results)
        context = "\n\n########\n\n".join(reranked_retrieved_docs)
        
        # Check if the query is asking for a summary
        is_summary = "summary" in query.lower()
        prompt = self._get_prompt(query, context, is_summary=is_summary)

        self.logger.info(f"-> User Query: {query}")
        self.logger.info(f"-> Context: {context[:500]}...")  # Truncate context for logging

        start_time = time.time()

        try:
            response, input_tokens, output_tokens = self.llm_inference.generate_text(
                prompt=prompt, model_name=ollama_model, llm_provider='Ollama'
            )
            if isinstance(response, dict) and "error" in response:
                self.logger.error(f"LLM returned an error: {response['error']}")
                return f"Error: {response['error']}", 0, self.n_results, 0, 0
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error generating response: {str(e)}", 0, self.n_results, 0, 0

        response_time = time.time() - start_time
        self.logger.info(f"-> LLM Response: {response[:500]}...")  # Truncate response for logging
        self.logger.info(f"-> Output token count: {output_tokens} | Input token count: {input_tokens} | Response time: {self._format_time(response_time)}")

        return response, self._format_time(response_time), self.n_results, input_tokens, output_tokens
    
    def generate_response2(self, query: str, llm_name='QwQ-32B', api_key=None):
        if not api_key and not API_KEY:
            self.logger.error("No API key provided for OpenRouter.")
            return "Set OpenRouter API Key", 0, 0, 0, 0
        
        api_key = api_key if api_key else API_KEY

        self.logger.info(f"--> Generate Response Using OpenRouter LLM: {llm_name}")
        stored_documents = db_utils.load_documents_from_db()
        if not stored_documents:
            self.logger.warning("No documents found in the database.")
            return "No documents available. Please upload a document first.", 0, 0, 0, 0

        chunks, embeddings = self._retrieve(query, stored_documents, n_results=50)

        if not chunks:
            self.logger.warning("No relevant chunks retrieved for the query.")
            return "No relevant information found. Please ensure documents are uploaded or rephrase your query.", 0, 0, 0, 0
        
        reranked_retrieved_docs = self._rerank_docs(chunks=chunks, embeddings=embeddings, query=query, top_k=self.n_results)
        context = "\n\n########\n\n".join(reranked_retrieved_docs)

        # Check if the query is asking for a summary
        is_summary = "summary" in query.lower()
        prompt = self._get_prompt(query, context, is_summary=is_summary)

        self.logger.info(f"-> Context: {context[:500]}...")  # Truncate context for logging

        start_time = time.time()

        try:
            response, input_tokens, output_tokens = self.llm_inference.generate_text(
                prompt=prompt, model_name=llm_name, llm_provider='Sambanova'
            )
            if isinstance(response, str) and response.startswith("API Error"):
                self.logger.error(f"API Error: {response}")
                return response, 0, self.n_results, 0, 0
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error generating response: {str(e)}", 0, self.n_results, 0, 0

        response_time = time.time() - start_time
        self.logger.info(f"-> LLM Response: {response[:500]}...")  # Truncate response for logging
        self.logger.info(f"-> Output token count: {output_tokens} | Input token count: {input_tokens} | Response time: {self._format_time(response_time)}")
        
        return response, self._format_time(response_time), self.n_results, input_tokens, output_tokens

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)
        self.logger.info(f"Deleted ChromaDB collection: {self.collection_name}")

if __name__ == "__main__":
    rag_system = RAGSystem(collection_name="pdf_content", db_path="PDF_ChromaDB", n_results=10)
    print(rag_system.generate_response2("What is the name of the book?", "QwQ-32B"))
