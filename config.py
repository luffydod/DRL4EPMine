from dataclasses import dataclass
import platform


@dataclass
class PPOConfig:
    # 环境相关配置
    env_id: str = "EpMineEnv-v0"
    num_envs: int = 1
    seed: int = 4399 # 224
    only_image: bool = True
    only_state: bool = False
    
    # PPO算法超参数
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 32
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # 训练相关配置
    total_timesteps: int = int(5e5)
    policy: str = "cnn"
    """ choose from 
        'cnn', -- 默认的CNN策略网络
        'mlp', -- 默认的MLP策略网络
        'resnet', -- 自定义的ResNet策略网络
        'cnn_custom' -- 自定义的CNN策略网络
        'cnn_pro' -- 改进NatureCNN策略网络
    """
    verbose: int = 1
    tensorboard_log: str = "tblogs"
    
    # 模型保存相关
    save_path: str = "models"
    save_freq: int = 100000
    
    # 环境相关配置
    file_name: str = "MineField_Windows-0510-random/drl.exe" if platform.system() == "Windows" else "MineField_Linux-0510-random/drl.x86_64"
    port: int = 30001
    work_id: int = 0
    time_scale: float = 20.0