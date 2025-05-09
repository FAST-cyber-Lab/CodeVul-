
import os
import zipfile
import gdown

def download_and_extract_models(file_id, output_dir="saved_models", zip_name="saved_models.zip"):

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, zip_name)

    if not os.path.exists(zip_path):
        print("Downloading model zip from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_path, quiet=False)
    else:
        print("Model zip already exists")

    print("Extracting zip file...")
    if zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Models extracted to: {output_dir}")
    else:
        print(f"The downloaded file '{zip_path}' is not a valid zip file")

