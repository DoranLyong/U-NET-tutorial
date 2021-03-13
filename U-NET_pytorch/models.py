#%%
import torch 
import torch.nn as nn  # 학습 가능한 레이어들을 담은 패키지 ; # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F # 학습 안 되는 레이어들을 담은 패키지 ; # All functions that don't have any parameters, relu, tanh, etc. 
import torchvision.transforms.functional as TF   # data transformation을 위한 (ref) https://pytorch.org/vision/stable/transforms.html#functional-transforms


# %%
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        """
        여기서는 학습이 가능한 계층을 설계한다. 
        예를 들어 nn 패키지를 활용하는 것들 
        
        (ex) nn.conv2d, nn.Linear, etc.
        """

        self.conv = nn.Sequential(  nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
                                    nn.BatchNorm2d(out_channels),  # U-NET 논문은 2015년 출판, BN 논문은 2016에 나왔지만...
                                    nn.ReLU(inplace=True),
                                    )

    def forward(self, x):
        """
        여기서는 학습 파라미터가 없는 계층을 설계한다. 
        예를 들어 nn.functional 패키지를 활용한 것들 
        
        (ex) F.relu, F.max_pool2d, etc.
        """
        return self.conv(x)

# %%
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512] ):
        super(UNET, self).__init__()

        self.ups = nn.ModuleList()  # nn.Module 을 list 형태로 정리 ; (ref) https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
                                    # 비슷한 역할을 하는 것이 nn.Sequential(), nn.ModuleDict(); (ref) https://greeksharifa.github.io/pytorch/2018/11/10/pytorch-usage-03-How-to-Use-PyTorch/
                                    # 언제언제 Sequential, ModuleList, ModuleDict을 사용해야할 까? ; (ref) https://towardsdatascience.com/pytorch-how-and-when-to-use-module-sequential-modulelist-and-moduledict-7a54597b5f17                                                                   
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


# %% 모델 확인 
if __name__ == "__main__":

    # 프로세스 장비 설정 
    gpu_no = 0
    device = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')

    # 모델 초기화 
    model = VGG_net(in_channels=3,num_classes=10).to(device)
    print(model)


    # 모델 출력 테스트 
    x = torch.randn(1, 3, 224, 224).to(device)
    print(model(x).shape)