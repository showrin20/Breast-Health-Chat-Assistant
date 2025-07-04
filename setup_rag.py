import os
import pickle
import faiss
from utils.rag import RAGSystem

def main():
    # Define the path to the documents directory
    documents_path = "rag/documents"
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag_system = RAGSystem()
    
    # Build index from documents
    print("Processing documents and building index...")
    rag_system.build_index(documents_path)
    
    # Save embeddings and index
    if rag_system.embeddings is not None and rag_system.index is not None:
        # Save document embeddings
        with open("rag/doc_embeddings.pkl", "wb") as f:
            pickle.dump(rag_system.embeddings, f)
        print("Saved document embeddings to rag/doc_embeddings.pkl")
        
        # Save FAISS index
        faiss.write_index(rag_system.index, "rag/vector_store.faiss")
        print("Saved FAISS index to rag/vector_store.faiss")
        
        print(f"RAG setup complete! Processed {len(rag_system.document_chunks)} document chunks.")
    else:
        print("Error: Failed to generate embeddings or index. Check document path and content.")

if __name__ == "__main__":
    main()