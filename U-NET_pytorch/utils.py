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



#%% Save the check point 
""" (ref) https://github.com/DoranLyong/DeepLearning-model-factory/blob/master/ML_tutorial/PyTorch/Basics/06_model_loadsave_CNN.py
"""
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print(f"=> {Back.RED}Saving checkpoint{Style.RESET_ALL}")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> {Back.RED}Loading checkpoint{Style.RESET_ALL}")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint['optimizer'])

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


#%%
def check_accuracy(loader, model, device="cuda"):
    """ (ref) https://github.com/DoranLyong/ResNet-tutorial/blob/main/ResNet_pytorch/ResNet18_for_CIFAR-10.py
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval() # eval mode

    with torch.no_grad():
        for x, y in loader: # 미니배치 별로 iteration 
            """ Get data to cuda if possible
            """
            x = x.to(device)
            y = y.to(device).unsqueeze(1) # gray scale 이미지이기 때문에 C 값은 없음 
                                          # [B, C=1, H, W] -> [B, H, W]

            """ forward 
            """
            preds = torch.sigmoid(model(x))  # Get probability map
            preds = (preds > 0.5).float() # 픽셀별로 확률 값이 >0.5 인 것만 True
                                          # (예시) [0.5, 0.3, 0.9] -> [False, False, True] -> [0., 0., 1.]
                                          # 본 예시는 binary segmentation을 다루고 있음 

            num_correct += (preds == y).sum() # 정답 개수 카운트 
            num_pixels += torch.numel(preds) # the number of elements; (ref) https://pytorch.org/docs/stable/generated/torch.numel.html

            dice_score += (2 * (preds * y).sum()) / (
                          (preds + y).sum() + 1e-8
                        )

    print( f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels * 100:.2f}%"    )
    print(f"Dice score: {dice_score/len(loader)}")

    model.train() # train mode again



#%% validation example output 
def save_predictions_as_imgs( loader, model, folder="saved_pred_images/", device="cuda", cur_epoch:int=0):

    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)

        with torch.no_grad():
            preds = torch.sigmoid(model(x)) # Get probability map
            preds = (preds > 0.5).float()  # 픽셀별로 확률 값이 >0.5 인 것만 True

        torchvision.utils.save_image(  preds, f"{folder}/pred_{idx}_{cur_epoch}epoch.png"   ) #(ref) https://pytorch.org/vision/stable/utils.html#torchvision.utils.save_image
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}_{cur_epoch}epoch.png")

    model.train()