"""
Download StatPearls Data from NCBI Bookshelf

According to MedRAG/StatPearls dataset documentation:
- StatPearls content cannot be directly distributed due to privacy policy
- We need to download raw data from NCBI Bookshelf
- Then process it using the chunking method from MedRAG

Reference: https://huggingface.co/datasets/MedRAG/statpearls
"""

import os
import tarfile
import urllib.request
from pathlib import Path


def download_statpearls_from_ncbi(
    output_dir: str = "./data/corpus/statpearls",
) -> None:
    """
    Download StatPearls from NCBI Bookshelf
    
    The dataset is available at:
    https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz
    """
    print("=" * 60)
    print("Downloading StatPearls from NCBI Bookshelf")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Download URL
    download_url = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"
    tar_gz_file = os.path.join(output_dir, "statpearls_NBK430685.tar.gz")
    
    # Check if already downloaded
    if os.path.exists(tar_gz_file):
        print(f"\n✓ Archive already exists: {tar_gz_file}")
    else:
        print(f"\nDownloading from: {download_url}")
        print("This may take a while (large file)...")
        
        try:
            # Download with progress
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                print(f"\rProgress: {percent:.1f}%", end="")
            
            urllib.request.urlretrieve(
                download_url,
                tar_gz_file,
                reporthook=report_progress
            )
            print(f"\n✓ Download complete: {tar_gz_file}")
            print(f"  Size: {os.path.getsize(tar_gz_file) / (1024*1024):.1f} MB")
        except Exception as e:
            print(f"\n✗ Error downloading: {e}")
            print("\nAlternative: Download manually from:")
            print(f"  {download_url}")
            return
    
    # Extract the archive
    print(f"\nExtracting archive...")
    extract_dir = os.path.join(output_dir, "raw")
    
    try:
        with tarfile.open(tar_gz_file, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print(f"✓ Extracted to: {extract_dir}")
    except Exception as e:
        print(f"✗ Error extracting: {e}")
        return
    
    # List extracted files
    print("\nExtracted files:")
    for root, dirs, files in os.walk(extract_dir):
        for file in files[:10]:  # Show first 10 files
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, extract_dir)
            print(f"  - {rel_path}")
    
    print(f"\nNext step: Process the markdown files using MedRAG chunking method")
    print("See: https://github.com/Teddy-XiongGZ/MedRAG/blob/main/src/data/statpearls.py")


def main():
    """Main function"""
    print("=" * 60)
    print("StatPearls Data Downloader (NCBI Bookshelf)")
    print("=" * 60)
    print("\nNote: StatPearls content is subject to privacy policy.")
    print("We download from official NCBI Bookshelf source.\n")
    
    download_statpearls_from_ncbi()
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
