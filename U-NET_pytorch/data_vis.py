#%%
import os.path as osp 
import random 

import numpy as np 
import cv2 

from dataset import CarvanaDataset


random.seed(42)

# %%
image_dir = osp.join("data", "train_images")
mask_dir = osp.join("data", "train_masks")



# ================================================================= #
#                        Load Data your dataset                     #
# ================================================================= #
# %% 커스텀 데이터 로드 
dataset = CarvanaDataset(   image_dir = image_dir,
                            mask_dir = mask_dir,
                            transform=None
                        )
# %% 이미지 & 마스크 가져오기 
idx = random.randrange(0, len(dataset))

image, mask = dataset[idx]

print(image.shape)
print(mask.shape)


# %% overlay segmented image on top of main image in python
""" (ref) https://stackoverflow.com/questions/57576686/how-to-overlay-segmented-image-on-top-of-main-image-in-python
"""
print(mask.max())


# %% Display 
cv2.imshow("img", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
