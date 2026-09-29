import os
import glob
from pathlib import Path
import nibabel as nib
import numpy as np
import src.config as config

def discover_patient_files():
    """
       Block 1: Locate and validate matching FLAIR and Segmentation NIfTI files.
    """
    patient_paths = sorted(glob.glob(str(config.DATA_DIR / "BraTS20_Training_*")))
    valid_patients = []
    skipped_patients = []

    for patient_dir in patient_paths:
        seg_files = glob.glob(os.path.join(patient_dir, "*_seg.nii*"))
        flair_files = glob.glob(os.path.join(patient_dir, "*_flair.nii*"))

        # Ensure exactly 1 FLAIR and 1 SEG file exist for this patient
        if len(seg_files) == 1 and len(flair_files) == 1:
            valid_patients.append({
                "patient_id": os.path.basename(patient_dir),
                "flair": flair_files[0],
                "seg": seg_files[0]
            })
        else:
            skipped_patients.append(patient_dir)

    print(f"[Block 1] Found {len(valid_patients)} valid patients. Skipped {len(skipped_patients)}.")
    return valid_patients




def load_and_normalize_volume(flair_path: str, seg_path: str):
    """
    Block 2: Load 3D NIfTI files and apply Z-score normalization 
    strictly on non-zero brain tissue.
    """
    # 1. Load NIfTI volumes into float32 NumPy arrays
    flair_nii = nib.load(flair_path).get_fdata().astype(np.float32)
    seg_nii = nib.load(seg_path).get_fdata().astype(np.float32)

    # 2. Sanity check spatial alignment
    assert flair_nii.shape == seg_nii.shape, f"Shape mismatch: {flair_nii.shape} vs {seg_nii.shape}"

    # 3. Create boolean mask for non-zero brain tissue
    brain_mask = flair_nii > 0

    # 4. Normalize FLAIR intensities inside brain region only
    if np.any(brain_mask):
        mean = np.mean(flair_nii[brain_mask])
        std = np.std(flair_nii[brain_mask])
        flair_nii[brain_mask] = (flair_nii[brain_mask] - mean) / (std + 1e-8)

    return flair_nii, seg_nii



def process_patient_slices(flair_nii: np.ndarray, seg_nii: np.ndarray):
    """
    Block 3: Slice 3D volume into 2D, filter empty slices, binarize masks,
    and format arrays for PyTorch (1, H, W).
    """
    processed_slices = []
    total_slices = seg_nii.shape[2]

    for slice_idx in range(total_slices):
        slice_flair = flair_nii[:, :, slice_idx]
        slice_seg = seg_nii[:, :, slice_idx]

        # Job 1: Filter pure air slices (must have sufficient brain tissue)
        has_brain_tissue = np.count_nonzero(slice_flair != 0) > 1000
        if not has_brain_tissue:
            continue

        # Job 2: Binarize mask (0 = background, 1 = tumor)
        binary_mask = (slice_seg > 0).astype(np.float32)

        # Job 3: Classification target for Multi-Task Learning
        has_tumor = int(np.count_nonzero(binary_mask) > 0)

        # Job 4: Keep tumor slices AND healthy brain slices in core depth range (30-125)
        # (This ensures negative controls for both seg and classification heads)
        if has_tumor or (30 <= slice_idx <= 125):
            # Shape transformation: (H, W) -> (1, H, W)
            image_tensor = np.expand_dims(slice_flair, axis=0)
            mask_tensor = np.expand_dims(binary_mask, axis=0)

            processed_slices.append({
                "slice_idx": slice_idx,
                "image": image_tensor,
                "mask": mask_tensor,
                "has_tumor": has_tumor
            })

    return processed_slices







def export_patient_slices(patient_id: str, processed_slices: list):
    """
    Block 4: Save processed slices to disk in compressed .npz format,
    organized by patient folders.
    """
    # 1. Create patient-specific output directory
    patient_output_dir = config.OUTPUT_DIR / patient_id
    patient_output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    # 2. Iterate through each processed slice and write to disk
    for slice_data in processed_slices:
        slice_idx = slice_data["slice_idx"]
        file_path = patient_output_dir / f"slice_{slice_idx:03d}.npz"

        np.savez_compressed(
            file_path,
            image=slice_data["image"],
            mask=slice_data["mask"],
            has_tumor=slice_data["has_tumor"]
        )
        saved_count += 1

    return saved_count



def run_preprocessing_pipeline():
    """
    Runs the complete pipeline for preprocessing BraTS training data.
    """
    # Block 1: Discovery
    valid_patients = discover_patient_files()

    total_slices_saved = 0

    for patient in valid_patients:
        patient_id = patient["patient_id"]

        # Block 2: Load & Normalize
        flair_nii, seg_nii = load_and_normalize_volume(patient["flair"], patient["seg"])

        # Block 3: Slice, Filter, & Format
        processed_slices = process_patient_slices(flair_nii, seg_nii)

        # Block 4: Export to Disk
        count = export_patient_slices(patient_id, processed_slices)
        total_slices_saved += count

        print(f"Patient {patient_id}: Exported {count} slices.")

    print(f"\n[Pipeline Complete] Exported {total_slices_saved} total slices across {len(valid_patients)} patients.")

if __name__ == "__main__":
    run_preprocessing_pipeline()