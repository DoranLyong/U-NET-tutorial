"""
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/utils.py

"""
#%%
import sys
import os 

import torch
import torchvision
from torch.utils.data import DataLoader # Gives easier dataset management and creates mini batches
from hydra import utils  # output/working directory (ref) https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/#internaldocs-banner
from colorama import Back, Style # 텍스트 컬러 출력 (ref) https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal


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
    print("*** | Data loading... | ***")
    """ Output/Working directory 
        (ref) https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/#internaldocs-banner
    """ 
    print(f"train_dir: {Back.GREEN} {utils.to_absolute_path(train_maskdir)} {Style.RESET_ALL}")
    print(f"train_maskdir: {Back.GREEN} {utils.to_absolute_path(train_maskdir)} {Style.RESET_ALL}")
    print(f"val_dir: {Back.MAGENTA} {utils.to_absolute_path(val_dir)} {Style.RESET_ALL}")
    print(f"val_maskdir: {Back.MAGENTA} {utils.to_absolute_path(val_maskdir)} {Style.RESET_ALL}")

    train_ds = CarvanaDataset(
        image_dir=utils.to_absolute_path(train_dir),
        mask_dir=utils.to_absolute_path(train_maskdir),
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
        image_dir=utils.to_absolute_path(val_dir),
        mask_dir=utils.to_absolute_path(val_maskdir),
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