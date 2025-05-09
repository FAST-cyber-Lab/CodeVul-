import os
import argparse
import pandas as pd
from utils.model_downloader import download_and_extract_models
from utils.classifier import predict_vulnerabilities

def main():
    parser = argparse.ArgumentParser(description="Predict code vulnerabilities using CodeVul+")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings CSV file")
    parser.add_argument("--output", type=str, default="vulnerability_predictions.csv", help="Path to save prediction results")
    parser.add_argument("--models_dir", type=str, default="saved_models/saved_models", help="Directory containing the saved models")
    parser.add_argument("--download", action="store_true", help="Download models before prediction")
    parser.add_argument("--file_id", type=str, default="1dQZOOY80DxYCtUZz962yQn-5JyuZvtNP", help="Google Drive file ID for models")
    
    args = parser.parse_args()
    
    if args.download:
        download_and_extract_models(args.file_id)
    
    if not os.path.exists(args.embeddings):
        print(f"Error: Embeddings file {args.embeddings} not found")
        return
    

    results = predict_vulnerabilities(
        embeddings_path=args.embeddings,
        models_dir=args.models_dir,
        output_path=args.output
    )
    

    if results is not None:
        vulnerable_count = results['prediction'].sum()
        total_count = len(results)
        print(f"\nPrediction Summary:")
        print(f"Total samples: {total_count}")
        print(f"Vulnerable samples: {vulnerable_count} ({vulnerable_count/total_count*100:.2f}%)")
        print(f"Non-vulnerable samples: {total_count-vulnerable_count} ({(total_count-vulnerable_count)/total_count*100:.2f}%)")
        print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main()
