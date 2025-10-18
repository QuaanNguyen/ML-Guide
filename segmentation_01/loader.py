import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import numpy as np

# --- 1. Custom Dataset Class for Segmentation01 ---

class SegmentationDataset(Dataset):
    """
    Custom Dataset for loading the Segmentation01 data (Image-Mask pairs).
    Loads original images (org) as RGB and masks (label) as LongTensor.
    """
    def __init__(self, root_dir, split, img_size=256):
        """
        Args:
            root_dir (str): Root directory (e.g., 'Segmentation01').
            split (str): 'train' or 'test'.
            img_size (int): The size for resizing the images and masks.
        """
        self.split = split
        self.org_dir = os.path.join(root_dir, split, 'org')
        self.label_dir = os.path.join(root_dir, split, 'label')
        
        list_file = os.path.join(root_dir, f'list_{split}.txt')
        
        # Read file names (which are assumed to be simple numbers, e.g., '1', '2')
        try:
            self.file_names = []
            with open(list_file, 'r', encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    image_path_str = parts[0]
                    normalized_path = image_path_str.replace('\\','/')
                    base_filename = os.path.basename(normalized_path)
                    name, ext = os.path.splitext(base_filename)
                    self.file_names.append(name)
        except FileNotFoundError:
             raise FileNotFoundError(f"List file not found: {list_file}")

        if not self.file_names:
            raise ValueError(f"No filenames found in {list_file}. Check file content.")

        # --- Transforms Setup ---
        # 1. Image Transform (RGB): Resizing, ToTensor, and Normalization
        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # Standard RGB normalization (use common ImageNet stats as a default)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 2. Mask Transform: Only resizing to ensure dimensions match the image
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST), # Use NEAREST for masks
        ])


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        img_path = os.path.join(self.org_dir, f'{name}.png')
        mask_path = os.path.join(self.label_dir, f'{name}.png')

        try:
            # Load original image as RGB (3 channels)
            image = Image.open(img_path).convert('RGB')
            # Load mask as single-channel grayscale ('L')
            mask = Image.open(mask_path).convert('L')
        except (FileNotFoundError, UnidentifiedImageError) as e:
            raise RuntimeError(f"Failed to load image or mask for '{name}': {e}")

        # Apply transforms
        image = self.img_tf(image)
        mask = self.mask_tf(mask)
        
        # Convert mask to a binary LongTensor
        mask_np = np.array(mask)
        # Normalize: [0, 255] -> [0, 1] (or whatever is background/foreground)
        # Assumes white (255) is the lung/foreground (1) and black (0) is background (0)
        mask_tensor = torch.from_numpy(mask_np // 255).long() 
        
        # Segmentation masks are typically [C=1, H, W] in PyTorch
        return image, mask_tensor.unsqueeze(0)


# --- 2. Helper function to Create DataLoaders ---

def make_segmentation_loaders(
    root_dir, img_size=256, batch_size=4, num_workers=8, pin_memory=True
):
    """
    Creates PyTorch DataLoaders for the Segmentation01 dataset.
    """
    train_ds = SegmentationDataset(root_dir, 'train', img_size)
    test_ds = SegmentationDataset(root_dir, 'test', img_size)

    # Note: On Windows, set num_workers=0 if you encounter multiprocessing errors.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    # Return 2 as the number of classes (background and foreground/lung)
    return train_loader, test_loader, 2
