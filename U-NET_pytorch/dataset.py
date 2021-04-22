"""
*                   (ref) https://youtu.be/IHq1t7NxS8k?t=1341
*                   (ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/dataset.py
* 커스텀 데이터 로더 (ref) https://github.com/DoranLyong/DeepLearning-model-factory/blob/master/ML_tutorial/PyTorch/Basics/07_custom_dataset_image.py
* Image aug 활용    (ref) https://github.com/DoranLyong/DeepLearning-model-factory/blob/master/ML_tutorial/PyTorch/Basics/Albumentations-image_augmentation_tutorial/02_segmentation.py
"""
#%%
import os.path as osp 
import os 



from PIL import Image  # torch normally handles images in PIL type before tensor
import numpy as np 
from torch.utils.data import Dataset  # 가져다쓸 데이터셋 객체를 지칭하는 클래스 (ref) https://huffon.github.io/2020/05/26/torch-data/




# ================================================================= #
#                 1. Create the Dataset Loader                      #
# ================================================================= #
# %% 1. 데이터셋 로더 객체 생성 
class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None ):
        """
        가져다쓸 데이터셋의 정보를 초기화한다. 
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform  # albumentation 활용? (ref) https://github.com/DoranLyong/DeepLearning-model-factory/blob/master/ML_tutorial/PyTorch/Basics/Albumentations-image_augmentation_tutorial/02_segmentation.py
        self.images = os.listdir(image_dir)  # os.listdir 대신에 사용할 것 (ref) https://m.blog.naver.com/hankrah/221826518915


    def __len__(self):
        """ (ref) https://dgkim5360.tistory.com/entry/python-duck-typing-and-protocols-why-is-len-built-in-function
            * 클래스의 맴버변수가 list 타입일 때
            * 그 길이를 반환함 
        """
        return len(self.images) # 이미지 목록 길이 반환 

    def __getitem__(self, index):
        """ (ref) https://jinmay.github.io/2019/11/26/python/python-instance-slice/
            * 클래스 맴버변수가 iterable 할 때 
            * 아이템 인덱싱해서 반환 
        """
        img_path = osp.join(self.image_dir, self.images[index]) # 가져올 이미지 인덱스의 Path 
        mask_path = osp.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif")) # 확장자 부분만 수정 

        image = np.array(Image.open(img_path).convert("RGB")) # 시각화/Image augmentation을 위해 Numpy 이미지 객체로 변환 
                                                              # (ref) https://m.blog.naver.com/nostresss12/221950215408
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)     
        mask[mask == 255.0] = 1.0  # get binary mask     

        
        if self.transform is not None:
            """ Mask augmentation for segmentation
                (ref) https://albumentations.ai/docs/getting_started/mask_augmentation/
            """
            augmentation = self.transform(image=image, mask=mask) 
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask 


if __name__ == '__main__':

    list_path = Path(".")
    print(list_path.name)
    print([i for i in Path(".").iterdir()])
# %%
