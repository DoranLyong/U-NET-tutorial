"""
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/utils.py

"""
#%%
import sys
import os 

import torch
import torchvision
from torch.utils.data import DataLoader # Gives easier dataset management and creates mini batches

from dataset import CarvanaDataset



#%% DataLoader 
""" (ref) https://github.com/DoranLyong/DeepLearning-model-factory/blob/master/ML_tutorial/PyTorch/Basics/07_custom_dataset_image.py
"""
def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    
):
    print(f"train_dir:{train_dir}")
    print(f"cwd: {os.getcwd()}")
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader