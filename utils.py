import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Optional

def display_images(categories: List[str], base_path: str = "datasets/MVTec") -> None:
    """
    Display one image from each category in a 3x5 grid.

    Args:
        categories: List of category names to display images from.
        base_path: The base path to the dataset.
    """
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()

    for idx, category in enumerate(categories):
        try:
            good_path = os.path.join(base_path, category, "train", "good")
            image_path = find_first_image(good_path)

            if image_path:
                image = load_and_convert_image(image_path)
                axes[idx].imshow(image)
                axes[idx].set_title(category)
            else:
                axes[idx].text(0.5, 0.5, f"No image found for {category}", 
                              ha='center', va='center')
            
            axes[idx].axis("off")
        except Exception as e:
            print(f"Error loading image for category {category}: {e}")
            axes[idx].text(0.5, 0.5, f"Error: {category}", ha='center', va='center')
            axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def display_defect_types(categories: List[str], base_path: str = "datasets/MVTec") -> None:
    """
    Display all defect types from the test folder for each category.
    
    Args:
        categories: List of category names to display defect images from.
        base_path: The base path to the dataset.
    """
    for category in categories:
        try:
            test_path = os.path.join(base_path, category, "test")
            defect_types = [d for d in os.listdir(test_path) if d != "good"]

            if not defect_types:
                print(f"No defect types found for category: {category}")
                continue
                
            print(f"Displaying defects for category: {category}")
            fig, axes = plt.subplots(1, len(defect_types), figsize=(len(defect_types) * 3, 5))
            
            # Ensure axes is always a list for consistent indexing
            if len(defect_types) == 1:
                axes = [axes]
            
            for idx, defect_type in enumerate(defect_types):
                defect_folder = os.path.join(test_path, defect_type)
                image_path = find_first_image(defect_folder)

                if image_path:
                    image = load_and_convert_image(image_path)
                    axes[idx].imshow(image)
                    axes[idx].set_title(defect_type)
                else:
                    axes[idx].text(0.5, 0.5, f"No image found for {defect_type}", 
                                  ha='center', va='center')
                
                axes[idx].axis("off")

            plt.suptitle(f"Defect Types for Category: {category}")
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)  # Adjust title placement
            plt.show()

        except Exception as e:
            print(f"Error loading defect images for category {category}: {e}")


def find_first_image(folder_path: str) -> Optional[str]:
    """
    Find the first image file (PNG or JPG) in the specified folder.
    
    Args:
        folder_path: Path to the folder to search for images.
        
    Returns:
        Path to the first image found, or None if no images are found.
    """
    for ext in ["*.png", "*.jpg"]:
        image_files = glob.glob(os.path.join(folder_path, ext))
        if image_files:
            return image_files[0]
    return None


def load_and_convert_image(image_path: str) -> Image.Image:
    """
    Load an image and convert it to RGB if it's grayscale.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        The loaded image in RGB format.
    """
    image = Image.open(image_path)
    
    # Convert grayscale images to RGB
    if image.mode == 'L':
        image = image.convert("RGB")
        
    return image
