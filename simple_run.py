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
    
    dataset_path = os.path.join(root_dir, 'data', 'mini_sample.csv')
    model_path = os.path.join(root_dir, 'pretrained', 'pretrained.pt')
    embeddings_path = os.path.join(root_dir, 'data', 'sample_embeddings.csv')
    predictions_path = os.path.join(root_dir, 'data', 'vulnerability_predictions.csv')

    
    train_script_path = os.path.join(root_dir, 'train.py')
    infer_script_path = os.path.join(root_dir, 'infer.py')
    predict_script_path = os.path.join(root_dir, 'predict.py')
    
    parser = argparse.ArgumentParser(description="Run CodeVul+ sample workflow")

    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code_vulplus_path = os.path.join(base_path, 'CodeVulplus')

    train_path = os.path.join(code_vulplus_path, 'train.py')
    infer_path = os.path.join(code_vulplus_path, 'infer.py')
    predict_path = os.path.join(code_vulplus_path, 'predict.py')
    
    parser.add_argument('--train', action='store_true', help='Fine-tuned model on sample dataset') 
    parser.add_argument('--infer', action='store_true', help='Generate embeddings from pretrained model')
    parser.add_argument('--predict', action='store_true', help='Run vulnerability prediction')
    parser.add_argument('--download_models', action='store_true', help='Download pre-trained classification models')
    parser.add_argument('--file_id', type=str, default="1dQZOOY80DxYCtUZz962yQn-5JyuZvtNP", 
                        help="Google Drive file ID for classification models")
    args = parser.parse_args()
    run_all = not (args.train or args.infer or args.predict or args.download_models)


    if not preprocess_sample_dataset(dataset_path):
        return

    if args.download_models or (run_all or args.predict):
        print("\n=== Downloading classification models ===")
        download_and_extract_models(args.file_id)
    
    if args.train or run_all:
        print("=== Training CodeVul+ model on sample dataset ===")
        print("Note: Code cleaning will be automatically applied to all samples")
        train_cmd = f"python {train_script_path} --dataset {dataset_path} --save_path {model_path} --batch_size 4 --epochs 2"
        print(f"Running: {train_cmd}")
        os.system(train_cmd)
    
    if args.infer or run_all:
        print("\n=== Generating embeddings using trained model ===")
        print("Note: Code cleaning will be automatically applied to all samples")
        infer_cmd = f"python {infer_script_path} --dataset {dataset_path} --model_path {model_path} --output {embeddings_path}"
        print(f"Running: {infer_cmd}")
        os.system(infer_cmd)
        
        if os.path.exists(embeddings_path):
            print(f"\nSuccess! Sample embeddings saved to {embeddings_path}")
            print("You can now use these embeddings for downstream tasks like vulnerability detection.")

    if args.predict or run_all:
        if os.path.exists(embeddings_path):
            print("\n=== Predicting vulnerabilities from embeddings ===")
            predict_cmd = f"python {predict_script_path} --embeddings {embeddings_path} --output {predictions_path}"
            print(f"Running: {predict_cmd}")
            os.system(predict_cmd)
        else:
            print(f"Error: Embeddings file not found at {embeddings_path}. Run inference first.")


    if run_all and os.path.exists(predictions_path):
        print("\n=== CodeVul+ Workflow Complete ===")
        print(f"1. Model trained and saved to: {model_path}")
        print(f"2. Code embeddings saved to: {embeddings_path}")
        print(f"3. Vulnerability predictions saved to: {predictions_path}")
        print("\nTo view prediction results: pd.read_csv('data/vulnerability_predictions.csv')")

if __name__ == "__main__":
    main()
