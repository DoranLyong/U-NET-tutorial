"""
Albumentations 활용 (ref) https://github.com/DoranLyong/DeepLearning-model-factory/blob/master/ML_tutorial/PyTorch/Basics/Albumentations-image_augmentation_tutorial/02_segmentation.py
Train 코드 블럭 참고 (ref) https://github.com/DoranLyong/VGG-tutorial/blob/main/VGG_pytorch/VGG_for_CIFAR10.py
Hydra 사용 하는 방법 (ref) https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b
본문 참고            (ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/train.py
"""

#%%


from tqdm import tqdm 
import yaml
import hydra   # for handling yaml (ref) https://neptune.ai/blog/how-to-track-hyperparameters
from omegaconf import DictConfig
import torch 
import torch.nn as nn  # 학습 가능한 레이어들을 담은 패키지 ; # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.backends.cudnn as cudnn    # https://hoya012.github.io/blog/reproducible_pytorch/
                                        # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
import torch.optim as optim # 최적화 알고리즘을 담은 패키지 ; # For all Optimization algorithms, SGD, Adam, etc.
import albumentations as A 
from albumentations.pytorch import ToTensorV2

from models import UNET
from utils import ( get_loaders,


                )



def train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE, BATCH_SIZE, NUM_EPOCHS,cur_epoch):

    # (ref) https://github.com/DoranLyong/DeepLearning-model-factory/blob/master/ML_tutorial/PyTorch/Basics/lr_scheduler_tutorial.py
    loop = tqdm(enumerate(train_loader), total=len(train_loader))  

    for batch_idx, (data, targets) in loop: # 미니배치 별로 iteration 
        """Get data to cuda if possible
        """
        data = data.to(device=DEVICE)  # 미니 베치 데이터를 device 에 로드 
        targets = targets.float().unsqueeze(1).to(device=DEVICE)    # 레이블 for supervised learning 
                                                                    # [B, 1, H, W] -> [B, H, W]

        """ Forward 
        """
        with torch.cuda.amp.autocast():
            """ Automatic Mixed Precision
                (ref) https://cvml.tistory.com/8
                (ref) https://debuggercafe.com/automatic-mixed-precision-training-for-deep-learning-using-pytorch/
            """
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        """ Backward 
        """     
        optimizer.zero_grad()   # AutoGrad 하기 전에(=역전파 실행전에) 매번 mini batch 별로 기울기 수치를 0으로 초기화        
        scaler.scale(loss).backward()        # (ref) https://tutorials.pytorch.kr/beginner/pytorch_with_examples.html
        scaler.step(optimizer)  # gradient descent or adam step
        scaler.update()         # weight update 

        """ Progress bar with tqdm
            (ref) https://github.com/DoranLyong/VGG-tutorial/blob/main/VGG_pytorch/VGG_for_CIFAR10.py
        """
        loop.set_description(f"Epoch [{cur_epoch}/{NUM_EPOCHS}], LR={ optimizer.param_groups[0]['lr'] :.1e}")

        if batch_idx % BATCH_SIZE == 0:  # 결과 표시 주기 
            loop.set_postfix( acc=(predictions == targets).sum().item() / predictions.size(0), loss=loss.item(),  batch=batch_idx)





@hydra.main(config_name='./cfg.yaml')
def main(cfg: DictConfig):
    """ Set your device 
    """
    gpu_no = 0  # gpu_number 
    DEVICE = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() and cfg.trainingInput.DEVICE.CUDA else 'cpu')
    print(f"device: { DEVICE }")


    """ Data augmentation params 
        (ref) https://github.com/DoranLyong/DeepLearning-model-factory/blob/master/ML_tutorial/PyTorch/Basics/Albumentations-image_augmentation_tutorial/04_cvt2pytorch.py
    """
    train_transform = A.Compose(
        [
            A.Resize(height=cfg.hyperparams.IMAGE_HEIGHT, width=cfg.hyperparams.IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0, # 분모 값이 최대 255 =>  uint8 이미지에 대해 정규화 
            ),
            ToTensorV2(), # Albumentations to torch.Tensor
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=cfg.hyperparams.IMAGE_HEIGHT, width=cfg.hyperparams.IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    """ Initialize the network 
    """
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    print(f"Model: {model}")

    model = torch.nn.DataParallel(model)# 데이터 병렬처리          # (ref) https://tutorials.pytorch.kr/beginner/blitz/data_parallel_tutorial.html
                                        # 속도가 더 빨라지진 않음   # (ref) https://tutorials.pytorch.kr/beginner/former_torchies/parallelism_tutorial.html
                                        # 오히려 느려질 수 있음    # (ref) https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b
    cudnn.benchmark = True


    """ Loss and optimizer  
    """
    loss_fn = nn.BCEWithLogitsLoss()    # Binary-CrossEntropy + sigmoid layer ; 클래스가 2개인 binary case에 대한 학습용 
                                        # (ref) https://nuguziii.github.io/dev/dev-002/
                                        # (ref) https://youtu.be/IHq1t7NxS8k?t=2199
    optimizer = optim.Adam(model.parameters(), lr=cfg.hyperparams.LEARNING_RATE)

    
    """ Get dataloader 
        (ref) https://youtu.be/IHq1t7NxS8k?t=2271
    """
    

    train_loader, val_loader = get_loaders( cfg.dataPath.TRAIN_IMG_DIR,  
                                            cfg.dataPath.TRAIN_MASK_DIR,
                                            cfg.dataPath.VAL_IMG_DIR,  
                                            cfg.dataPath.VAL_MASK_DIR,  
                                            cfg.hyperparams.BATCH_SIZE,
                                            train_transform, 
                                            val_transforms,                                            
                                            cfg.trainingInput.NUM_WORKERS,
                                            cfg.trainingInput.PIN_MEMORY,
                                        )

    """ Gradient Scaling
        (ref) https://pytorch.org/docs/stable/amp.html#gradient-scaling
    """        
    scaler = torch.cuda.amp.GradScaler()      



    """ Start the training-loop
    """                  
    for epoch in range(cfg.hyperparams.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, 
                loss_fn, scaler, DEVICE, 
                cfg.hyperparams.BATCH_SIZE, 
                cfg.hyperparams.NUM_EPOCHS, 
                epoch
                )
    


if __name__ == "__main__":



    main()     


# %%
