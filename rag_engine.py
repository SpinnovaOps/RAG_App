import os
import tiktoken
import numpy as np
from typing import List, Dict, Tuple, Any
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai
from dotenv import load_dotenv

from document_processor import Document, Chunk

# Load environment variables
load_dotenv()

# Set up Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Determine which model to use
try:
    available_models = genai.list_models()
    gemini_models = [m.name for m in available_models if "gemini" in m.name.lower()]
    
    if any("gemini-1.5-pro" in m for m in gemini_models):
        DEFAULT_MODEL = next(m for m in gemini_models if "gemini-1.5-pro" in m)
    elif any("gemini-pro" in m for m in gemini_models):
        DEFAULT_MODEL = next(m for m in gemini_models if "gemini-pro" in m)
    elif gemini_models:
        DEFAULT_MODEL = gemini_models[0]
    else:
        DEFAULT_MODEL = "models/gemini-1.5-pro"  # Default fallback
    
    print(f"Using model: {DEFAULT_MODEL}")
except Exception as e:
    print(f"Error listing models: {e}")
    DEFAULT_MODEL = "models/gemini-1.5-pro"  # Default fallback

class RAGEngine:
    """Retrieval-Augmented Generation engine using Gemini."""
    
    def __init__(self):
        """Initialize the RAG engine."""
        self.documents = {}  # Map document_id to Document object
        self.embeddings = {}  # Map chunk_id to embedding vector
        self.vectorizers = {}  # Map document_id to TF-IDF vectorizer
    
    def add_document(self, document: Document, embeddings: Dict[str, np.ndarray], vectorizer: TfidfVectorizer):
        """Add a document and its embeddings to the engine."""
        self.documents[document.id] = document
        self.embeddings.update(embeddings)
        self.vectorizers[document.id] = vectorizer
    
    def get_document(self, document_id: str) -> Document:
        """Get a document by ID."""
        return self.documents.get(document_id)
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents as dictionaries."""
        return [doc.to_dict() for doc in self.documents.values()]
    
    def search(self, query: str, document_id: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Search for relevant chunks using TF-IDF similarity."""
        document = self.get_document(document_id)
        if not document:
            return []
        
        # Get the vectorizer for this document
        vectorizer = self.vectorizers.get(document_id)
        if not vectorizer:
            return []
        
        # Transform the query using the document's vectorizer
        query_vector = vectorizer.transform([query]).toarray()[0]
        
        # Normalize the query vector
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        # Calculate similarity between query and all chunks
        results = []
        for chunk in document.chunks:
            chunk_embedding = self.embeddings.get(chunk.id)
            if chunk_embedding is None:
                continue
            
            similarity = 1 - cosine(query_vector, chunk_embedding)
            results.append((chunk, similarity))
        
        # Sort by similarity (highest first) and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    def generate_answer(self, query: str, contexts: List[Tuple[Chunk, float]], max_tokens: int = 1000) -> Tuple[str, int]:
        """Generate an answer using Gemini with retrieved contexts."""
        # Create the prompt with contexts
        context_text = "\n\n".join([f"Context {i+1}:\n{chunk.text}" for i, (chunk, _) in enumerate(contexts)])
        
        prompt = f"""
You are a helpful assistant answering questions based on the provided document.
Use ONLY the following context to answer the question. If you don't know the answer based on the context, say you don't know.

{context_text}

Question: {query}

Answer:"""

        try:
            # Generate response using Gemini
            model = genai.GenerativeModel(DEFAULT_MODEL)
            generation_config = genai.types.GenerationConfig(
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_output_tokens=1024,
            )
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            answer = response.text
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            # Simple fallback: extract relevant sentences from contexts
            answer = "I couldn't generate a proper response. Here are some relevant extracts:\n\n"
            for i, (chunk, _) in enumerate(contexts[:3]):
                # Take the first 200 characters of each chunk as a preview
                answer += f"Extract {i+1}: {chunk.text[:200]}...\n\n"
            answer += "(Note: This is a simplified response due to API limitations.)"
        
        # Estimate token count
        token_count = self.count_tokens(prompt) + self.count_tokens(answer)
        
        return answer, token_count
    
    def generate_summary(self, document_id: str, length: str = "medium") -> Tuple[str, int]:
        """Generate a summary of the document using Gemini."""
        document = self.get_document(document_id)
        if not document:
            return "Document not found", 0
        
        # Determine summary length guidance
        length_descriptions = {
            "short": "a brief summary (around 100 words)",
            "medium": "a comprehensive summary (around 200 words)",
            "long": "a detailed summary (around 400 words)"
        }
        length_description = length_descriptions.get(length, "a comprehensive summary (around 200 words)")
        
        # Prepare the document content
        content = document.content
        
        # Truncate content if it's too long
        max_content_tokens = 3000  # Limit context to 3000 tokens
        if self.count_tokens(content) > max_content_tokens:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(content)
            content = encoding.decode(tokens[:max_content_tokens])
            content += "\n[Document truncated due to length]"
        
        prompt = f"Please provide {length_description} of the following document:\n\n{content}"
        
        try:
            # Generate the summary using Gemini
            model = genai.GenerativeModel(DEFAULT_MODEL)
            generation_config = genai.types.GenerationConfig(
                temperature=0.3,
                top_p=0.95,
                top_k=40,
                max_output_tokens=1024,
            )
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            summary = response.text
            
        except Exception as e:
            print(f"Gemini API error during summarization: {e}")
            # Fallback: Extract first few paragraphs
            paragraphs = content.split('\n\n')
            summary = '\n\n'.join(paragraphs[:3])
            summary += "\n\n(Note: This is an extracted preview due to API limitations.)"
        
        # Estimate token count
        token_count = self.count_tokens(prompt) + self.count_tokens(summary)
        
        return summary, token_count
    
    def generate_questions(self, document_id: str, num_questions: int = 5) -> List[str]:
        """Generate suggested questions for a document using Gemini."""
        document = self.get_document(document_id)
        if not document:
            return []
        
        # Sample the document if it's too large
        content = document.content
        max_content_tokens = 3000
        if self.count_tokens(content) > max_content_tokens:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(content)
            content = encoding.decode(tokens[:max_content_tokens])
            content += "\n[Document truncated due to length]"
        
        # Create prompt for generating questions
        prompt = f"""
        You are analyzing a document to generate insightful questions that could be asked about it.
        Review the following document content and generate {num_questions} distinct, substantive questions 
        that explore key concepts, facts, or implications found in the text.
        
        Make sure your questions:
        1. Are specific to the document content
        2. Cover different aspects of the document
        3. Range from factual to analytical/interpretive
        4. Would be interesting and useful for someone trying to understand the document
        
        Document content:
        {content}
        
        Generate exactly {num_questions} questions in a numbered list. Each question should be on a single line without any additional explanation.
        """
        
        try:
            # Generate questions using Gemini
            model = genai.GenerativeModel(DEFAULT_MODEL)
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,  # Higher temperature for more creative questions
                top_p=0.95,
                top_k=40,
                max_output_tokens=1024,
            )
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Process response to extract just the questions
            questions_text = response.text
            
            # Parse the numbered list
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                # Check if line starts with a number followed by a period or parenthesis
                if line and (line[0].isdigit() or 
                            (len(line) > 1 and line[0].isdigit() and line[1] in ['.', ')', ']'])):
                    # Extract just the question part, removing the number prefix
                    question = line.split(' ', 1)[1] if ' ' in line else line
                    # Remove any leading special characters
                    while question and not question[0].isalnum():
                        question = question[1:]
                    questions.append(question)
            
            # Ensure we return the requested number of questions
            if len(questions) < num_questions:
                # If we didn't get enough questions, add generic ones based on document title/content
                for i in range(len(questions), num_questions):
                    questions.append(f"What is the significance of {document.title if hasattr(document, 'title') else 'this document'}?")
            
            # Limit to requested number
            return questions[:num_questions]
            
        except Exception as e:
            print(f"Gemini API error during question generation: {e}")
            # Fallback: Generate generic questions
            title = getattr(document, 'title', 'the document')
            return [
                f"What is the main purpose of {title}?",
                f"What are the key points covered in {title}?",
                f"How does {title} relate to the broader context?",
                f"What conclusions can be drawn from {title}?",
                f"What questions remain unanswered by {title}?"
            ][:num_questions]
    
    def analyze_document(self, document_id: str) -> Dict[str, Any]:
        """Perform a comprehensive analysis of a document."""
        document = self.get_document(document_id)
        if not document:
            return {"error": "Document not found"}
        
        # Generate summary
        summary, _ = self.generate_summary(document_id)
        
        # Generate sample questions
        questions = self.generate_questions(document_id)
        
        # Get document metadata
        metadata = {
            "id": document.id,
            "title": getattr(document, 'title', 'Untitled'),
            "chunk_count": len(document.chunks),
            "word_count": len(document.content.split()),
            "estimated_token_count": self.count_tokens(document.content)
        }
        
        # Return comprehensive analysis
        return {
            "metadata": metadata,
            "summary": summary,
            "suggested_questions": questions
        }