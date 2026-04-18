import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class CareerChatbot:
    def __init__(self, data_path=None):
        print("Initializing Career Chatbot Component...")
        local_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
        os.makedirs(local_cache, exist_ok=True)
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=local_cache)
        # Load small generative model
        print("Loading LLM model (google/flan-t5-small)...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", cache_dir=local_cache)
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", cache_dir=local_cache)
        
        self.chunks = []
        self.index = None
        
        if data_path and os.path.exists(data_path):
            self._build_index(data_path)
            
    def _build_index(self, data_path):
        print(f"Building FAISS index from {data_path}...")
        with open(data_path, 'r') as f:
            text = f.read()
            
        # VERY simple sentence/paragraph chunking
        paragraphs = text.split('\n\n')
        for p in paragraphs:
            sentences = p.split('. ')
            self.chunks.extend([s.strip() for s in sentences if len(s.strip()) > 10])
            
        print(f"Generated {len(self.chunks)} knowledge chunks.")
        
        # Embed and index
        embeddings = self.embedding_model.encode(self.chunks)
        
        # Initialize FAISS Index
        d = embeddings.shape[1] # Dimension of embeddings
        self.index = faiss.IndexFlatL2(d)
        self.index.add(np.array(embeddings).astype('float32'))
        print("FAISS Index build complete.")

    def get_most_relevant(self, query, top_k=2):
        if not self.index:
            return ["No knowledge base available."]
            
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        relevant_chunks = []
        for i in indices[0]:
            if i != -1 and i < len(self.chunks):
                relevant_chunks.append(self.chunks[i])
                
        return relevant_chunks
        
    def ask_question(self, query):
        print(f"Chatbot processing query: '{query}'")
        
        # 1. Retrieve Context
        context_list = self.get_most_relevant(query)
        context = " \\n".join(context_list)
        
        # 2. Construct Prompt (Flan T5 instruction format)
        prompt = f"Answer the question based only on the following context.\\nContext: {context}\\nQuestion: {query}\\nAnswer:"
        
        # 3. Generate response using HuggingFace Pipeline
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.gen_model.generate(**inputs, max_new_tokens=150)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"LLM Generation error: {e}")
            answer = "I'm having trouble analyzing the knowledge base right now."
            
        return {
            "answer": answer,
            "context_used": context_list
        }

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, 'career_knowledge.txt')
    bot = CareerChatbot(data_path=data_file)
    print(bot.ask_question("How can I improve placement chances in data science?"))
