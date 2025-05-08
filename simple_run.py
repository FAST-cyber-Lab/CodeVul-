import os
import argparse
import pandas as pd
from utils.data_utils import clean_code

def preprocess_sample_dataset(dataset_path):

    print(f"Checking sample dataset at {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"Sample dataset not found at {dataset_path}. Please create it first.")
        return False
    
    try:

        df = pd.read_csv(dataset_path)
        
        if 'functionSource' not in df.columns:
            print("ERROR: Dataset must contain a 'functionSource' column")
            return False
            
        print(f"Dataset loaded successfully with {len(df)} samples.")
        return True
        
    except Exception as e:
        print(f"Error processing sample dataset: {str(e)}")
        return False

def main():

    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset_path = os.path.join(root_dir, 'data', 'sample_dataset.csv')
    model_path = os.path.join(root_dir, 'pretrained', 'pretrained.pt')
    embeddings_path = os.path.join(root_dir, 'data', 'sample_embeddings.csv')

    
    train_script_path = os.path.join(root_dir, 'CodeVulplus', 'train.py')
    infer_script_path = os.path.join(root_dir, 'CodeVulplus', 'infer.py')
    
    parser = argparse.ArgumentParser(description="Run CodeVul+ sample workflow")
    parser.add_argument('--train', default=train_script_path, help='Fine-tuned model on sample dataset')
    parser.add_argument('--infer', default=infer_script_path, help='Generate embeddings from fpretrained model')
    args = parser.parse_args()

    if not preprocess_sample_dataset(dataset_path):
        return
    
    if args.train or (not args.train and not args.infer):
        print("=== Training CodeVul+ model on sample dataset ===")
        print("Note: Code cleaning will be automatically applied to all samples")
        train_cmd = f"python train.py --dataset {dataset_path} --save_path {model_path} --batch_size 4 --epochs 2"
        print(f"Running: {train_cmd}")
        os.system(train_cmd)
    
    if args.infer or (not args.train and not args.infer):
        print("\n=== Generating embeddings using trained model ===")
        print("Note: Code cleaning will be automatically applied to all samples")
        infer_cmd = f"python infer.py --dataset {dataset_path} --model_path {model_path} --output {embeddings_path}"
        print(f"Running: {infer_cmd}")
        os.system(infer_cmd)
        
        if os.path.exists(embeddings_path):
            print(f"\nSuccess! Sample embeddings saved to {embeddings_path}")
            print("You can now use these embeddings for downstream tasks like vulnerability detection.")

if __name__ == "__main__":
    main()
