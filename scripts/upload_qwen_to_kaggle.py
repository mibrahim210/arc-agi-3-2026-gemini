"""Download Qwen2.5-Coder-7B GGUF model and upload as a Kaggle dataset.

Usage:
    .venv\\Scripts\\python.exe scripts/upload_qwen_to_kaggle.py
"""
import os
import sys
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "scratch" / "qwen25_coder_7b_gguf"
GGUF_URL = "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
GGUF_FILENAME = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"

def download_file(url: str, dest_path: Path):
    print(f"Downloading {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest_path), reporthook=report_progress)
    print("\nDownload complete!")

def main():
    gguf_path = DATASET_DIR / GGUF_FILENAME
    if not gguf_path.exists():
        download_file(GGUF_URL, gguf_path)
    else:
        print(f"File already exists at {gguf_path}")

    # Create dataset-metadata.json
    metadata = {
        "title": "qwen25-coder-7b-gguf",
        "id": "plasmacoder210/qwen25-coder-7b-gguf",
        "licenses": [{"name": "CC0-1.0"}]
    }
    metadata_path = DATASET_DIR / "dataset-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote metadata to {metadata_path}")

    print("\nTo upload to Kaggle, run:")
    print(f"  kaggle datasets create -p \"{DATASET_DIR}\"")

if __name__ == "__main__":
    main()
