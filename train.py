import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import os
import sys
from sklearn.model_selection import train_test_split


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import HybridModel, build_dynamic_graph
from utils import CodeEmbedder

def train_and_evaluate(dataset_path, save_path="models/codevul_plus.pt", batch_size=8, epochs=3, learning_rate=1e-4, val_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, dataset_path)
    save_path = os.path.join(script_dir, save_path)
    

    dataset = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(dataset)} samples")
    

    train_data, val_data = train_test_split(dataset, test_size=val_split, random_state=42)
    train_codes = train_data['functionSource'].tolist()
    val_codes = val_data['functionSource'].tolist()

    embedder = CodeEmbedder(device=device)
    

    token_lengths = [len(embedder.tokenizer.encode(code, truncation=False)) for code in train_codes[:min(100, len(train_codes))] + val_codes[:min(100, len(val_codes))]]
    optimal_max_length = int(np.percentile(token_lengths, 95))
    optimal_max_length = min(optimal_max_length, embedder.tokenizer.model_max_length)
    print(f"Optimal max_length: {optimal_max_length}")

    model_hybrid = HybridModel(768, 512, 768).to(device)
    optimizer = optim.Adam(model_hybrid.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()


    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    for epoch in range(epochs):

        model_hybrid.train()
        total_train_loss = 0.0
        
        for i in range(0, len(train_codes), batch_size):
            batch = train_codes[i:i + batch_size]
            embedded_batch = embedder.embed_code_snippets(batch, max_length=optimal_max_length)
            batch_graphs = [build_dynamic_graph(code, sequence_length=embedded_batch.shape[1], device=device) for code in batch]
            batch_graphs = torch.stack(batch_graphs)
            
            optimizer.zero_grad()
            outputs = model_hybrid(embedded_batch, batch_graphs)
            aggregated_outputs = torch.mean(outputs, dim=1)
            
            loss = loss_fn(aggregated_outputs, torch.mean(embedded_batch, dim=1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / (len(train_codes) / batch_size)
        
        model_hybrid.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for i in range(0, len(val_codes), batch_size):
                batch = val_codes[i:i + batch_size]
                embedded_batch = embedder.embed_code_snippets(batch, max_length=optimal_max_length)
                batch_graphs = [build_dynamic_graph(code, sequence_length=embedded_batch.shape[1], device=device) for code in batch]
                batch_graphs = torch.stack(batch_graphs)
                
                outputs = model_hybrid(embedded_batch, batch_graphs)
                aggregated_outputs = torch.mean(outputs, dim=1)
                
                val_loss = loss_fn(aggregated_outputs, torch.mean(embedded_batch, dim=1))
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / (len(val_codes) / batch_size)
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")
    
    torch.save(model_hybrid.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CodeVul+ model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV file")
    parser.add_argument("--save_path", type=str, default="models/codevul_plus.pt", help="Path to save trained model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    train_and_evaluate(
        dataset_path=args.dataset,
        save_path=args.save_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
