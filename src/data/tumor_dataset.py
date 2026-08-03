import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

class BraTS2DDataset(Dataset):
    """
    Dataset loader for BraTS 2020 3D MRI volumes, sliced into 2D images.
    Returns:
        image: Tensor of shape [1, H, W] (FLAIR modality normalized)
        mask:  Tensor of shape [1, H, W] (Binary mask: 1 for tumor, 0 for background)
    """
    def __init__(self, slice_indices: list, transform=None):
        """
        Args:
            slice_indices: List of tuples (patient_dir_path, slice_idx)
            transform: Optional torchvision transforms
        """
        self.slice_indices = slice_indices
        self.transform = transform

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        patient_path, slice_idx = self.slice_indices[idx]
        
        # Paths to MRI volumes (using FLAIR modality for baseline)
        folder_name = os.path.basename(patient_path)
        flair_path = os.path.join(patient_path, f"{folder_name}_flair.nii.gz")
        seg_path = os.path.join(patient_path, f"{folder_name}_seg.nii.gz")

        # Load 3D volumes using nibabel
        flair_vol = nib.load(flair_path).get_fdata()
        seg_vol = nib.load(seg_path).get_fdata()

        # Extract 2D Slice
        image_slice = flair_vol[:, :, slice_idx]
        mask_slice = seg_vol[:, :, slice_idx]

        # Convert multi-class segmentation mask to binary (tumor vs background)
        mask_slice = np.where(mask_slice > 0, 1.0, 0.0)

        # Intensity Normalization (Min-Max scaling to [0, 1])
        if np.max(image_slice) > 0:
            image_slice = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice))

        # Convert to PyTorch Tensors [C, H, W]
        image_tensor = torch.from_numpy(image_slice).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_slice).float().unsqueeze(0)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, mask_tensor