import torch
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
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
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
        
# 图像格式：[H, W, C]
class ResNetFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 冻结ResNet的所有参数
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # 移除最后的全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 添加一个新的全连接层，将ResNet的输出映射到所需的特征维度
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, features_dim),  # ResNet18的输出特征维度是512
            nn.ReLU()
        )
        
    def forward(self, observations):
        # 转置输入以匹配PyTorch的期望格式
        observations_channels_first = observations.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return self.linear(self.resnet(observations_channels_first))
    
class ResNetPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=ResNetFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            **kwargs
        )