# Replace ENTIRE file with this:

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

def generate_embeddings():
    """Generate embeddings using Hugging Face model"""
    
    print("🚀 Starting embedding generation with Hugging Face...")
    
    # Load the model
    print("📦 Loading Sentence-Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded!")
    
    # Load careers CSV
# In generate_embeddings.py line 21:
    csv_path = "../data/careers_updated_with_embeddings_text.csv"
    print(f"📂 Loading careers from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} careers")
    
    # Generate embeddings
    embeddings = []
    for idx, row in df.iterrows():
        embedding_text = row['embedding_text']
        
        # Generate embedding
        embedding = model.encode(embedding_text, convert_to_numpy=True)
        embeddings.append(embedding)
        
        # Progress
        if (idx + 1) % 10 == 0:
            print(f"   ✅ Processed {idx + 1}/{len(df)} careers")
    
    # Add embeddings to dataframe
    df['career_vector'] = embeddings
    
    # Save to pickle
    output_path = "careers_final_with_embeddings.pkl"
    df.to_pickle(output_path)
    print(f"✅ Saved embeddings to {output_path}")
    
    print(f"✨ Complete! Generated {len(df)} embeddings")
    print(f"📊 Embedding dimension: {len(embeddings[0])}")

if __name__ == "__main__":
    generate_embeddings()