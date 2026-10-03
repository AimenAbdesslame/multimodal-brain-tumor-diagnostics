import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTS2DDataset(Dataset):
    """Flexible PyTorch Dataset for loading pre-extracted 2D .npz slices.

    Accepts either a directory path (str) or an explicit list of .npz file
    paths.
    """

    def __init__(
        self, data_source, transform=None, cache_in_ram: bool = False
    ):
        self.transform = transform
        self.cache_in_ram = cache_in_ram

        if isinstance(data_source, list):
            self.slice_paths = sorted(data_source)
        elif isinstance(data_source, str):
            self.slice_paths = sorted(
                glob.glob(os.path.join(data_source, "*.npz"))
            )
        else:
            raise TypeError(
                f"Expected data_source to be a list or str, got {type(data_source)}"
            )

        if not self.slice_paths:
            raise FileNotFoundError(
                f"No .npz files found for input: {data_source}"
            )

        self.ram_cache = []
        if self.cache_in_ram:
            for path in self.slice_paths:
                with np.load(path) as data:
                    has_tumor = (
                        data["has_tumor"]
                        if "has_tumor" in data
                        else float((data["mask"] > 0).sum() > 0)
                    )
                    self.ram_cache.append(
                        (data["image"], data["mask"], has_tumor)
                    )

    def __len__(self) -> int:
        return len(self.slice_paths)

    def __getitem__(self, idx: int):
        if self.cache_in_ram:
            image, mask, has_tumor = self.ram_cache[idx]
        else:
            file_path = self.slice_paths[idx]
            with np.load(file_path) as data:
                image = data["image"]  # Shape: (1, H, W)
                mask = data["mask"]  # Shape: (1, H, W) or (H, W)
                has_tumor = (
                    float(data["has_tumor"])
                    if "has_tumor" in data
                    else float((mask > 0).sum() > 0)
                )

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).float()
        label_tensor = torch.tensor([has_tumor], dtype=torch.float32)

        if self.transform:
            augmented = self.transform(image=image_tensor, mask=mask_tensor)
            image_tensor, mask_tensor = augmented["image"], augmented["mask"]

        return image_tensor, mask_tensor, label_tensor