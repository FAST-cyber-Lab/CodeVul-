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
        # Define class names mapping
        class_names = {
            0: 'CWE-119',
            1: 'CWE-120',
            2: 'CWE-469',
            3: 'CWE-476',
            4: 'CWE-other'
        }
        
        # Count predictions for each class
        prediction_counts = results['prediction'].value_counts().sort_index()
        total_count = len(results)
        
        print(f"\nPrediction Summary:")
        print(f"Total samples: {total_count}")
        
        # Display counts and percentages for each vulnerability class
        print("\nVulnerability Class Distribution:")
        for class_idx, count in prediction_counts.items():
            if class_idx in class_names:
                class_name = class_names[class_idx]
                percentage = (count / total_count) * 100
                print(f"{class_name}: {count} samples ({percentage:.2f}%)")
        
        # Check for missing classes in the prediction results
        for class_idx, class_name in class_names.items():
            if class_idx not in prediction_counts.index:
                print(f"{class_name}: 0 samples (0.00%)")
        
        #print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main()
