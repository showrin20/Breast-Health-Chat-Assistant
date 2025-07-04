import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import PyPDF2
import re

class RAGSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
        self.document_chunks = []
        self.embeddings = None
        self.index = None
        
    def load_documents(self, documents_path: str) -> List[str]:
        """Load and process documents from the documents directory"""
        chunks = []
        
        if not os.path.exists(documents_path):
            return chunks
        
        for filename in os.listdir(documents_path):
            filepath = os.path.join(documents_path, filename)
            
            if filename.endswith('.pdf'):
                chunks.extend(self._process_pdf(filepath))
            elif filename.endswith('.txt'):
                chunks.extend(self._process_txt(filepath))
        
        return chunks
    
    def _process_pdf(self, filepath: str) -> List[str]:
        """Extract text from PDF and split into chunks"""
        chunks = []
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                # Split into chunks
                chunks = self._split_text(text)
        except Exception as e:
            print(f"Error processing PDF {filepath}: {e}")
        
        return chunks
    
    def _process_txt(self, filepath: str) -> List[str]:
        """Process text file and split into chunks"""
        chunks = []
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                chunks = self._split_text(text)
        except Exception as e:
            print(f"Error processing text file {filepath}: {e}")
        
        return chunks
    
    def _split_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into overlapping chunks"""
        # Clean the text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def build_index(self, documents_path: str):
        """Build the FAISS index from documents"""
        # Load documents
        self.document_chunks = self.load_documents(documents_path)
        
        if not self.document_chunks:
            print("No documents found. Adding default medical information...")
            self.document_chunks = self._get_default_medical_info()
        
        # Generate embeddings
        self.embeddings = self.embedding_model.encode(self.document_chunks)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Built index with {len(self.document_chunks)} document chunks")
    
    def _get_default_medical_info(self) -> List[str]:
        """Provide default medical information if no documents are available"""
        return [
            "Breast cancer screening typically involves mammograms, which are X-ray examinations of the breast. Regular screening can help detect cancer early when it's most treatable.",
            
            "A breast lump can be caused by many things, including cysts, fibroadenomas, or normal breast tissue changes. Most breast lumps are not cancer, but any new lump should be evaluated by a healthcare provider.",
            
            "Family history of breast cancer can increase your risk, but having a family member with breast cancer doesn't mean you will develop it. Genetic counseling can help assess your risk.",
            
            "Mammograms are highly effective screening tools, with accuracy rates of about 85-90% for detecting breast cancer. However, they may sometimes miss cancers or show false positives.",
            
            "Warning signs of breast cancer include a new lump or mass, breast swelling, skin irritation or dimpling, nipple discharge, and changes in breast size or shape.",
            
            "Self-breast examinations should be performed monthly, ideally a few days after your menstrual period ends when breasts are least tender and swollen.",
            
            "Risk factors for breast cancer include age, gender, family history, genetic mutations, personal history of breast cancer, and lifestyle factors like alcohol consumption and obesity.",
            
            "Treatment options for breast cancer vary depending on the type and stage but may include surgery, chemotherapy, radiation therapy, hormone therapy, and targeted therapy.",
            
            "Regular exercise, maintaining a healthy weight, limiting alcohol consumption, and avoiding smoking can help reduce breast cancer risk.",
            
            "Dense breast tissue can make mammograms more difficult to read and may slightly increase breast cancer risk. Additional screening methods may be recommended."
        ]
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Search for relevant document chunks"""
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return relevant chunks
        relevant_chunks = [self.document_chunks[i] for i in indices[0]]
        return relevant_chunks
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response using the context"""
        # Combine context
        context_text = "\n".join(context)
        
        # Create prompt
        prompt = f"""Context: {context_text}

Question: {query}

Please provide a warm, supportive, and informative answer based on the context. Use simple language and be encouraging. Always remind the user that this is general information and they should consult healthcare professionals for medical advice.

Answer:"""
        
        try:
            response = self.qa_pipeline(prompt, max_length=300, temperature=0.7)[0]['generated_text']
            
            # Clean up the response
            response = response.replace(prompt, "").strip()
            
            # Add supportive closing
            if not response.endswith(('.', '!', '?')):
                response += "."
            
            response += "\n\n💙 Remember, you're taking great care of your health by asking these questions. Always consult with your healthcare provider for personalized medical advice."
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble generating a response right now. Please try rephrasing your question or consult with a healthcare professional."

def initialize_rag(documents_path: str = "rag/documents") -> Optional[RAGSystem]:
    try:
        rag_system = RAGSystem()
        rag_system.build_index(documents_path)
        return rag_system
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return None

def get_rag_response(rag_system: RAGSystem, query: str) -> str:
    if rag_system is None:
        return "I'm sorry, the knowledge system is not available right now. Please try again later."
    
    try:
        # Search for relevant context
        relevant_chunks = rag_system.search(query)
        
        # Generate response
        response = rag_system.generate_response(query, relevant_chunks)
        
        return response
        
    except Exception as e:
        print(f"Error getting RAG response: {e}")
        return "I'm sorry, I encountered an error while processing your question. Please try again or contact support."