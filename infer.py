import torch
import numpy as np
import pandas as pd
import argparse
import os
import re

from models import HybridModel, build_dynamic_graph
from utils import CodeEmbedder
from utils.data_utils import clean_code

def load_trained_embeddings(dataset_path, model_path="pretrained/pretrained.pt", batch_size=8, output_path="embeddings.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    batch_codes = dataset['functionSource'].tolist()
    
    embedder = CodeEmbedder(device=device)
    
    model_hybrid = HybridModel(768, 512, 768).to(device)
    model_hybrid.load_state_dict(torch.load(model_path, map_location=device))
    model_hybrid.eval()
    
    output_list = []
    
    for i in range(0, len(batch_codes), batch_size):
        batch = batch_codes[i:i + batch_size]

        embedded_batch = embedder.embed_code_snippets(batch)
        batch_graphs = [build_dynamic_graph(clean_code(code), sequence_length=embedded_batch.shape[1], device=device) for code in batch]
        batch_graphs = torch.stack(batch_graphs)
        
        with torch.no_grad():
            outputs = model_hybrid(embedded_batch, batch_graphs)
            aggregated_outputs = torch.mean(outputs, dim=1)
            output_list.append(aggregated_outputs.cpu().numpy())
    
    output_array = np.vstack(output_list)
    
    feature_names = [f'gn{i+1}' for i in range(output_array.shape[1])]
    output_df = pd.DataFrame(output_array, columns=feature_names)
    
    output_df.to_csv(output_path, index=False)
    print(f"Embeddings saved to {output_path}")
    
    return output_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings using trained CodeVul+ model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV file")
    parser.add_argument("--model_path", type=str, default="pretrained/pretrained.pt", help="Path to trained model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--output", type=str, default="embeddings.csv", help="Path to save embeddings")
    
    args = parser.parse_args()
    
    load_trained_embeddings(
        dataset_path=args.dataset,
        model_path=args.model_path,
        batch_size=args.batch_size,
        output_path=args.output
    )
