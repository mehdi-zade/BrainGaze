import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class EEGSaliencyDataset(Dataset):
    """Dataset that aligns preprocessed EEG data with SALICON saliency maps in a unified structure.
    """

    def __init__(self, eeg_dir, stimulus_root, maps_root, split="training", transform=None):
        super().__init__()
        self.eeg_dir = eeg_dir
        self.stim_root = stimulus_root
        self.maps_root = maps_root
        self.split = split  # "training" or "test" (mapped to "val" in downstream scripts)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        suffix = "training" if split == "training" else "test"
        
        # Walk subject subfolders to find matched files
        eeg_files = []
        parquet_files = []
        for root, dirs, files in os.walk(eeg_dir):
            for f in files:
                if f.endswith(f'_eeg_{suffix}_matched.npy'):
                    eeg_files.append(os.path.join(root, f))
                elif f.endswith(f'_metadata_categories_{suffix}_matched.parquet'):
                    parquet_files.append(os.path.join(root, f))
                    
        if not eeg_files:
            raise FileNotFoundError(f"No matched EEG numpy files found under: {eeg_dir} for split: {split}")
            
        # Ensure they are sorted so they align
        eeg_files.sort()
        parquet_files.sort()
        
        print(f"Loading {len(eeg_files)} subject matched datasets...")
        eeg_arrays = []
        meta_dfs = []
        
        for eeg_path, pq_path in zip(eeg_files, parquet_files):
            # Load EEG data dict
            eeg_obj = np.load(eeg_path, allow_pickle=True)
            if isinstance(eeg_obj, np.ndarray) and eeg_obj.dtype == object:
                eeg_dict = eeg_obj.item()
            else:
                eeg_dict = eeg_obj
            
            # Load metadata parquet
            df = pd.read_parquet(pq_path)
            
            eeg_data = eeg_dict["preprocessed_eeg_data"].astype(np.float32)
            
            if len(df) != len(eeg_data):
                print(f"Warning: size mismatch in {eeg_path} and {pq_path}. Truncating.")
                min_len = min(len(df), len(eeg_data))
                df = df.iloc[:min_len]
                eeg_data = eeg_data[:min_len]
                
            eeg_arrays.append(eeg_data)
            meta_dfs.append(df)
            
        self.eeg_data = np.concatenate(eeg_arrays, axis=0)
        # Scale to µV for consistency
        self.eeg_data *= 1e6
        self.meta_df = pd.concat(meta_dfs, axis=0).reset_index(drop=True)
        
        print(f"Dataset ready – {len(self.meta_df)} samples aligned with SALICON maps.")

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        eeg_array = self.eeg_data[idx]  # Shape (32, 250)
        eeg_tensor = torch.from_numpy(eeg_array)
        
        row = self.meta_df.iloc[idx]
        map_filename = os.path.basename(row["map_path"])
        map_path = os.path.join(self.maps_root, map_filename)
        coco_id = int(row["coco_id"])
        
        # Load saliency map
        saliency_img = Image.open(map_path).convert('L')
        saliency_tensor = self.transform(saliency_img)
        if saliency_tensor.sum() > 0:
            saliency_tensor = saliency_tensor / saliency_tensor.sum()
            
        # Find and load corresponding stimulus image in flat folder
        stim_path = None
        for split_dir in ["train", "val"]:
            stim_name = f"COCO_{split_dir}2014_{coco_id:012d}.jpg"
            p = os.path.join(self.stim_root, stim_name)
            if os.path.exists(p):
                stim_path = p
                break
                
        if stim_path:
            stim_img = Image.open(stim_path).convert('RGB')
            stim_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            stim_tensor = stim_transform(stim_img)
        else:
            stim_tensor = torch.zeros(3, 224, 224)
            
        # Extract zero-indexed subject ID (e.g. "Subject 1" -> 0)
        subject_str = str(row["subject"])
        subject_id = int(subject_str.replace("Subject", "").strip()) - 1
            
        return {
            "eeg": eeg_tensor,            # (32, 250)
            "saliency": saliency_tensor, # (1, 224, 224)
            "image": stim_tensor,        # (3, 224, 224)
            "coco_id": coco_id,
            "subject_id": subject_id
        }
