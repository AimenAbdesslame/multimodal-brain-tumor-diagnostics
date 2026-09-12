import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class BraTS2DDataset(Dataset):
    """
    High-performance PyTorch Dataset loading pre-extracted 2D .npz slice files.
    Eliminates NIfTI decompressing bottlenecks for instant batch retrieval.
    """
    def __init__(self, data_dir: str, transform=None, cache_in_ram: bool = False):
        self.data_dir = data_dir
        self.transform = transform
        self.cache_in_ram = cache_in_ram
        
        # Collect all pre-extracted slice files
        self.slice_paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
        if not self.slice_paths:
            raise FileNotFoundError(f"No .npz files found in directory: {data_dir}")

        # Optional: Load all slices into RAM if dataset size permits (~2-4 GB)
        self.ram_cache = []
        if self.cache_in_ram:
            print(f"Caching {len(self.slice_paths)} slices directly into RAM...")
            for path in self.slice_paths:
                with np.load(path) as data:
                    self.ram_cache.append((data["image"], data["mask"]))

    def __len__(self) -> int:
        return len(self.slice_paths)

    def __getitem__(self, idx: int):
        if self.cache_in_ram:
            image, mask = self.ram_cache[idx]
        else:
            file_path = self.slice_paths[idx]
            with np.load(file_path) as data:
                image = data["image"]  # Expected shape: (4, H, W) for 4 MRI sequences
                mask = data["mask"]    # Expected shape: (H, W) for multi-class target

        # Convert numpy arrays to native PyTorch Tensors
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).long()

        if self.transform:
            augmented = self.transform(image=image_tensor, mask=mask_tensor)
            image_tensor, mask_tensor = augmented["image"], augmented["mask"]

        return image_tensor, mask_tensor