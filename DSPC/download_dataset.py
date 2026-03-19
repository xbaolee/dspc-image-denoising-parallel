from datasets import load_dataset
from PIL import Image
import os
import sys

def download_dataset_images():
    # Create folders
    os.makedirs("input_images", exist_ok=True)
    os.makedirs("clean_images", exist_ok=True)
    os.makedirs("output_images", exist_ok=True)
    
    print("Downloading datasets (this may take some time)...")
    
    # Load clean and noisy images
    clean_ds = load_dataset("Tsomaros/ImageNet-C-brightness-severity_1", split="validation[:70]")
    noisy_ds = load_dataset("Tsomaros/ImageNet-C-gaussian_noise-severity_3", split="validation[:70]")
    
    # Define resolutions
    resolutions = [(244, 244), (512, 512), (1024, 1024), (2048, 2048), (3840,2160),(5120, 2880),(7680, 4320)]

    # Process and save images
    for i in range(10):  # Using 10 images
        print(f"Processing image {i+1}/10")
        for width, height in resolutions:
            res_str = f"{width}x{height}"
            print(f"  Resizing to {res_str}")
            
            # Convert to grayscale and resize
            noisy = noisy_ds[i]["image"].convert("L").resize((width, height))
            clean = clean_ds[i]["image"].convert("L").resize((width, height))
            
            # Save images
            noisy.save(f"input_images/image_{i}_{res_str}.jpg")
            clean.save(f"clean_images/image_{i}_{res_str}.jpg")
    
    print("Image download complete!")
    print(f"Images saved to:")
    print(f"- {os.path.abspath('input_images')}")
    print(f"- {os.path.abspath('clean_images')}")

if __name__ == "__main__":
    download_dataset_images()