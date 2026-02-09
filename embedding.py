import torch
from sentence_transformers import SentenceTransformer

def generate_embeddings(sentences, model_name='all-MiniLM-L6-v2'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name)
    model.to(device)
    
    embeddings = model.encode(sentences, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    return embeddings