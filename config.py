from dataclasses import dataclass
import platform


@dataclass
class PPOConfig:
    # 环境相关配置
    env_id: str = "EpMineEnv-v0"
    num_envs: int = 1
    seed: int = 0
    
    # PPO算法超参数
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # 训练相关配置
    total_timesteps: int = int(1e4)
    policy: str = "CnnPolicy"
    verbose: int = 1
    
    # 模型保存相关
    save_path: str = "models"
    save_freq: int = 10000
    log_path: str = "logs"
    
    # 环境相关配置
    file_name: str = "MineField_Windows-0510-random/drl.exe" if platform.system() == "Windows" else "MineField_Linux-0510-random/drl.x86_64"
    port: int = 30001
    work_id: int = 0
    time_scale: float = 20.0