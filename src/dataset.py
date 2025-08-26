import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import random
import json
from datetime import datetime

# -------------------------------
# 1. Dataset Class
# -------------------------------
class ChestXrayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing patient_id, image_path, and labels
            img_dir (str): Directory with images
            transform: torchvision transforms for preprocessing & augmentation
        """
        self.data = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")

        labels = np.array(row[2:], dtype=np.float32)  # assuming labels start from 3rd col
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels)

# -------------------------------
# 2. Patient-wise Split + Logging
# -------------------------------
def patient_wise_split(df, seed=42, train_ratio=0.7, val_ratio=0.1, log_file="split_log.json"):
    patients = df["patient_id"].unique()
    random.seed(seed)
    random.shuffle(patients)

    n = len(patients)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]

    train_df = df[df["patient_id"].isin(train_patients)]
    val_df = df[df["patient_id"].isin(val_patients)]
    test_df = df[df["patient_id"].isin(test_patients)]

    # Logging split information
    split_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "num_patients": len(patients),
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
        "test_patients": len(test_patients),
        "train_patient_ids": train_patients.tolist(),
        "val_patient_ids": val_patients.tolist(),
        "test_patient_ids": test_patients.tolist()
    }

    with open(log_file, "w") as f:
        json.dump(split_info, f, indent=4)

    print(f"[LOG] Data splits saved to {log_file}")

    return train_df, val_df, test_df

# -------------------------------
# 3. Transforms (Preprocessing + Augmentation)
# -------------------------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# -------------------------------
# 4. Prepare Datasets & Loaders
# -------------------------------
csv_path = "labels.csv"   # labels CSV with [patient_id, image_path, labels...]
img_dir = "images/"       # directory of CXR images
labels_df = pd.read_csv(csv_path)

train_df, val_df, test_df = patient_wise_split(labels_df, seed=42)

train_dataset = ChestXrayDataset(train_df, img_dir, transform=train_transform)
val_dataset = ChestXrayDataset(val_df, img_dir, transform=val_test_transform)
test_dataset = ChestXrayDataset(test_df, img_dir, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
