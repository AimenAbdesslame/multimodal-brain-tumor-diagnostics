import torch 
import nibabel as nib
import numpy as np
import os 
from torch.utils.data import Dataset , DataLoader

class BraTS2DDataset(Dataset):
    def __init__(self , slice_indexes) :
        self.slice_indexes = slice_indexes 
        
    def __len__(self):
        return len(self.slice_indexes)
    
    def __getitem__(self , idx) : 
        patient_path , slice_idx = self.slice_indexes[idx] 
        folder_name = os.path.basename(patient_path)
        seg_path = os.path.join(patient_path, f"{folder_name}_seg.nii")
        flair_path = os.path.join(patient_path, f"{folder_name}_flair.nii")
        
        seg_img = nib.load(seg_path).get_fdata()[: , : , slice_idx]
        seg_binary = np.where(seg_img > 0 , 1.0 , 0.0) 
        seg_tensor = np.expand_dims(seg_binary , axis = 0 ) #shape  = 1 , 240 , 240 
        
        flair_img = nib.load(flair_path).get_fdata()[: , : , slice_idx]
        if flair_img.max() > 0 : 
            flair_img = (flair_img - flair_img.min() ) / (flair_img.max() - flair_img.min())
        flair_tensor = np.expand_dims(flair_img , axis = 0 ) # shape = 1 , 240 , 240
        
        return torch.from_numpy(flair_tensor).float(), torch.from_numpy(seg_tensor).float()
        
        