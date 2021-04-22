"""
Data spliting
(ref) https://doranlyong-ai.tistory.com/42?category=895489
"""
#%%
from pathlib import Path 
import os.path as osp
import os
from glob import glob
import shutil  # sh 명령어를 사용할 수 있는 유틸리티 (ref) https://code.tutsplus.com/ko/tutorials/file-and-directory-operations-using-python--cms-25817

from tqdm import tqdm 
import numpy as np


np.random.seed(42)

#%%
cwd = os.getcwd()
Img_dir = osp.join(cwd, 'data', 'train')
Mask_dir = osp.join(cwd, 'data', 'train_masks')

DATAESTS = [ 'train_set', 'val_set' ]


#%%
def get_names(path_list):
    name_list = [i.split('/')[-1].split('.')[0] for i in path_list]
    

    return sorted(name_list)



#%%

def run_split(Img_dir, Mask_dir):
    Img_list = sorted(glob(Img_dir + "/*.jpg"))
    Mask_list = sorted(glob(Mask_dir + "/*_mask.gif"))

    name_list = get_names(Img_list)

    


    """ Data split
    """
    item_names = np.array(name_list)
    np.random.shuffle(item_names)  # sorted 된 경로 정보를 뒤섞기 ; (ref) https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html

    img_split = np.split( 
                            item_names,
                            indices_or_sections = [ int( 0.8 * len(item_names) ),  int( 1.0 * len(item_names) )   ]   # 8:2 로 쪼갬 
                        )                                                                                           # (ref) https://www.w3resource.com/numpy/manipulation/split.php

    dataset_data = zip(DATAESTS, img_split) # {'train': train_paths, 'val': cal_paths }


    """ Move data 
    """
    for dst, items in dataset_data:
        IMG_DIR = Path(f'data/{dst}/images')
        MASK_DIR = Path(f'data/{dst}/masks')
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        MASK_DIR.mkdir(parents=True, exist_ok=True)


        loop = tqdm(items, total=len(items))
        for item in loop:
            img_src = osp.join(Img_dir, item + '.jpg' )  # source path
            mask_src = osp.join(Mask_dir, item + '_mask.gif' )

            img_dst = f'{cwd}/data/{dst}/images'   # destination (목적지)
            mask_dst = f'{cwd}/data/{dst}/masks'

            shutil.copy(img_src, img_dst)
            shutil.copy(mask_src, mask_dst)








#%%
if __name__ == '__main__':

    DATA_DIR = Path('data')
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    assert osp.isdir(osp.join(cwd, 'data', 'train')), 'Download the dataset first from the link: https://www.kaggle.com/c/carvana-image-masking-challenge/data '




    run_split(Img_dir, Mask_dir)


# %%
