import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Optional

DEFAULT_BASE_PATH = "datasets/MVTec"
SUPPORTED_EXTENSIONS = ["png", "jpg"]

def find_first_image(folder_path: str) -> Optional[str]:
    """
    Find the first image file in the specified folder.
    
    Args:
        folder_path: Path to the folder to search for images.
        
    Returns:
        Path to the first image found, or None if no images are found.
    """
    for ext in SUPPORTED_EXTENSIONS:
        image_files = glob.glob(os.path.join(folder_path, f"*.{ext}"))
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
    try:
        image = Image.open(image_path)
        # Convert grayscale images to RGB
        if image.mode == 'L':
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise IOError(f"Failed to load image {image_path}: {e}")

def find_corresponding_mask(image_path: Optional[str], gt_folder: str) -> Optional[str]:
    """
    Find corresponding ground truth mask given an image path.
    
    Args:
        image_path: Path to the original image.
        gt_folder: Folder containing ground truth masks.
        
    Returns:
        Path to corresponding mask or None.
    """
    if not image_path:
        return None
    
    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]
    mask_filename = f"{base_name}_mask.png"
    mask_path = os.path.join(gt_folder, mask_filename)
    
    return mask_path if os.path.exists(mask_path) else None

def display_images(categories: List[str], base_path: str = DEFAULT_BASE_PATH) -> None:
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

def display_defect_types(categories: List[str], base_path: str = DEFAULT_BASE_PATH) -> None:
    """
    Display all defect types from the test folder for each category.
    
    Args:
        categories: List of category names to display defect images from.
        base_path: The base path to the dataset.
    """
    for category in categories:
        try:
            test_path = os.path.join(base_path, category, "test")
            
            # Skip if test path doesn't exist
            if not os.path.exists(test_path):
                print(f"Test path not found for category: {category}")
                continue
                
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
            
        except FileNotFoundError:
            print(f"Directory not found for category {category}")
        except Exception as e:
            print(f"Error loading defect images for category {category}: {e}")

def visualize_defect_examples(
    categories: List[str], 
    base_path: str = DEFAULT_BASE_PATH, 
    alpha: float = 0.5
) -> None:
    """
    Visualize one original image and its ground truth mask side-by-side for each defect type.
    
    Args:
        categories: List of category names to visualize.
        base_path: Base path to the dataset.
        alpha: Transparency factor for mask overlay (0.0 to 1.0).
    """
    for category in categories:
        test_path = os.path.join(base_path, category, "test")
        gt_path = os.path.join(base_path, category, "ground_truth")
        
        # Check if paths exist
        if not os.path.exists(test_path) or not os.path.exists(gt_path):
            print(f"Test or ground truth path not found for category: {category}")
            continue
            
        defect_types = [d for d in os.listdir(test_path) if d != "good"]
        
        if not defect_types:
            print(f"No defects found for category: {category}")
            continue
            
        for defect in defect_types:
            defect_folder = os.path.join(test_path, defect)
            gt_defect_folder = os.path.join(gt_path, defect)
            
            if not os.path.exists(gt_defect_folder):
                print(f"Ground truth folder not found for {category}/{defect}")
                continue
                
            image_file = find_first_image(defect_folder)
            mask_file = find_corresponding_mask(image_file, gt_defect_folder)
            
            if image_file and mask_file:
                try:
                    original_img = cv2.imread(image_file)
                    mask_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    
                    # Convert original image from BGR to RGB
                    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    
                    # Create colored mask overlay (red)
                    mask_colored = np.zeros_like(original_img)
                    mask_colored[mask_img > 0] = [0, 0, 255]
                    
                    # Blend images
                    blended_img = cv2.addWeighted(
                        original_rgb, 1, 
                        cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB), 
                        alpha, 0
                    )
                    
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(original_rgb)
                    axes[0].set_title(f"{category} - {defect}\nOriginal")
                    axes[0].axis('off')
                    
                    axes[1].imshow(blended_img)
                    axes[1].set_title(f"{category} - {defect}\nWith Ground Truth Mask")
                    axes[1].axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Error processing images for {category}/{defect}: {e}")
            else:
                print(f"Missing image or mask for {category}/{defect}")
