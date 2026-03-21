import os
import io
import zipfile
import requests

def download_zip(url):
    """Download zip from URL and return as BytesIO."""
    response = requests.get(url)
    response.raise_for_status()
    return io.BytesIO(response.content)

def extract_zip(zip_file, extract_to):
    """Extract a zip file."""
    with zipfile.ZipFile(zip_file) as z:
        z.extractall(extract_to)

def extract_nested_zips(base_dir):
    """Find and extract nested zip files while preserving structure."""
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".zip"):
                zip_path = os.path.join(root, file)
                extract_dir = os.path.join(root, os.path.splitext(file)[0])

                os.makedirs(extract_dir, exist_ok=True)

                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(extract_dir)

                print(f"Extracted: {zip_path} -> {extract_dir}")

def main():
    url = ""
    output_dir = "output"

    os.makedirs(output_dir, exist_ok=True)

    # Download outer zip
    zip_bytes = download_zip(url)

    # Extract outer zip
    extract_zip(zip_bytes, output_dir)

    # Extract inner zips
    extract_nested_zips(output_dir)

    print("Done.")

if __name__ == "__main__":
    main()
