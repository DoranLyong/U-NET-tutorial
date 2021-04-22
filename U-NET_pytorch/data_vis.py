#%%
import os.path as osp 
import random 
from typing import Tuple

import numpy as np 
import cv2 
import matplotlib.pyplot as plt

from dataset import CarvanaDataset


random.seed(42)

# %%
image_dir = osp.join("data", "val_set", "images")
mask_dir = osp.join("data", "val_set","masks")



# ================================================================= #
#                        Load Data your dataset                     #
# ================================================================= #
# %% 커스텀 데이터 로드 
dataset = CarvanaDataset(   image_dir = image_dir,
                            mask_dir = mask_dir,
                            transform=None
                        )
# %% 이미지 & 마스크 가져오기 
idx = random.randrange(0, len(dataset))  # __len__ 메소드 호출 

image, mask = dataset[idx]  # __getitem__ 메소드 호출 
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

print(image.shape)
print(mask.shape)

# %% overlay segmented image on top of main image in python
""" Method1 
    (ref) https://stackoverflow.com/questions/57576686/how-to-overlay-segmented-image-on-top-of-main-image-in-python
"""


#%%

""" Method2 : Crop 
    (ref) https://stackoverflow.com/questions/64076134/how-to-display-a-ground-truth-image-segmentation-mask-image-in-python
"""
masked = image.copy()
masked[mask == 0] = 0 
cv2.imshow("(Method2) masked", masked)


""" Method3 : Color overlay 
    (ref) https://stackoverflow.com/questions/9193603/applying-a-coloured-overlay-to-an-image-in-either-pil-or-imagemagik
    (ref) https://www.kaggle.com/purplejester/showing-samples-with-segmentation-mask-overlay
"""
color: Tuple[int, int, int] = (255, 0, 0)
alpha: float = 0.5

mask_anno = np.where(mask == 1) # booelan mask 

out = image.copy()
img_layer = image.copy()
img_layer[mask_anno] = color # broadcasting the color values in each channel
out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)

cv2.imshow("(Method3) color overlayed", out)


""" Method4 : Contour with color overlay 
    (ref) https://www.programmersought.com/article/33965071602/
    (ref) https://www.aiuai.cn/aifarm276.html
"""


# %% Display 
cv2.imshow("img", image)
cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
