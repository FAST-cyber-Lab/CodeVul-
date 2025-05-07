import os
import argparse
import sys

def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_path = os.path.join('data', 'sample_dataset.csv')
    model_path = os.path.join('models', 'codevul_plus_sample.pt')
    embeddings_path = os.path.join('data', 'sample_embeddings.csv')
    
    parser = argparse.ArgumentParser(description="Run CodeVul+ sample workflow")
    parser.add_argument('--train', action='store_true', help='Train model on sample dataset')
    parser.add_argument('--infer', action='store_true', help='Generate embeddings from trained model')
    args = parser.parse_args()

    original_dir = os.getcwd()
    os.chdir(script_dir)
    
    try:
        if args.train or (not args.train and not args.infer):
            print("=== Training CodeVul+ model on sample dataset ===")
            train_cmd = f"{sys.executable} train.py --dataset {dataset_path} --save_path {model_path} --batch_size 4 --epochs 2"
            print(f"Running: {train_cmd}")
            os.system(train_cmd)
        
        if args.infer or (not args.train and not args.infer):
            print("\n=== Generating embeddings using trained model ===")
            infer_cmd = f"{sys.executable} infer.py --dataset {dataset_path} --model_path {model_path} --output {embeddings_path}"
            print(f"Running: {infer_cmd}")
            os.system(infer_cmd)
            
            if os.path.exists(embeddings_path):
                print(f"\nSuccess! Sample embeddings saved to {embeddings_path}")
                print("You can now use these embeddings for downstream tasks like vulnerability detection.")
    finally:

        os.chdir(original_dir)

if __name__ == "__main__":
    main()
