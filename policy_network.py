import torch as th
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy

# 图像格式：[C, H, W]
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # 改进的CNN网络 - 更深层次但保持小卷积核
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算CNN输出特征维度
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        # 更复杂的MLP头部
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        return self.linear(self.cnn(observations))

# 自定义策略
class CustomCnnPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
            **kwargs
        )
        
class ResNetFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # 将所有BatchNorm换成GroupNorm
        for m in self.resnet.modules():
            if isinstance(m, nn.BatchNorm2d):
                m = nn.GroupNorm(32, m.num_features)

        # 移除最后的全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1], nn.Flatten())

    def forward(self, observations):
        return self.resnet(observations)

class ResNetPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=ResNetFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            **kwargs
        )
# v1
# class NatureCNNpro(BaseFeaturesExtractor):
    
#     def __init__(self,observation_space, features_dim: int = 512) -> None:
#         super().__init__(observation_space, features_dim)
        
#         # C = 3
#         n_input_channels = observation_space.shape[0]
        
#         # 使用更复杂的CNN架构，包括更多层、残差连接和批归一化
#         self.cnn = nn.Sequential(
#             # 第一层卷积块
#             nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
#             nn.ReLU(),
            
#             # 第二层卷积块
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
            
#             # 第三层卷积块
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
            
#             # 第四层卷积块
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
            
#             # 展平层
#             nn.Flatten(),
#         )
        
#         # 计算展平后的特征数量
#         with th.no_grad():
#             n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
#         # 使用多层感知机处理展平后的特征
#         self.linear = nn.Sequential(
#             nn.Linear(n_flatten, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, features_dim),
#             nn.ReLU()
#         )
    
#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         return self.linear(self.cnn(observations)) 


# v2 (better)
class NatureCNNpro(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]  # 3通道RGB图像
        
        self.cnn = nn.Sequential(
            # 第一层：使用较小的卷积核和步长
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            
            # 第二层
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 第三层
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 展平层
            nn.Flatten(),
        )
        
        # 计算展平后的特征数量
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        # 简化全连接层
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class NatureCnnproPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=NatureCNNpro,
            features_extractor_kwargs=dict(features_dim=128),
            **kwargs
        )
        
class SimpleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        
        # 添加下采样路径
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class SimpleResNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        # 减小初始卷积核大小，降低参数量
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.relu = nn.ReLU(inplace=True)
        
        # 调整残差块结构
        self.layer1 = SimpleResBlock(32, 64, stride=1)
        self.layer2 = SimpleResBlock(64, 96, stride=2)
        self.layer3 = SimpleResBlock(96, 128, stride=2)
        
        # 计算最终特征图大小
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            x = self.conv1(sample)
            x = self.gn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            self.final_flatten_size = x.flatten(1).shape[1]
        
        # 改进全连接层，添加dropout防止过拟合
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        x = self.conv1(observations)
        x = self.gn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.fc(x)
        
        return x

class SimpleResNetPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=SimpleResNetExtractor,
            features_extractor_kwargs=dict(features_dim=512),  # 增加特征维度
            **kwargs
        )
        