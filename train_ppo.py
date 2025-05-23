import os
import argparse
import torch as th
import torch.nn as nn
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from envs.SingleAgent.mine_toy import EpMineEnv

from config import PPOConfig

# 图像格式：[H, W, C]
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # 获取图像尺寸
        n_input_channels = observation_space.shape[2]  # 通道在最后
        
        # CNN网络
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算CNN输出特征维度
        with th.no_grad():
            # 注意：这里我们需要转置输入，因为PyTorch期望通道优先格式
            sample = th.as_tensor(observation_space.sample()[None]).float()
            sample_channels_first = sample.permute(0, 3, 1, 2)  # NHWC -> NCHW
            n_flatten = self.cnn(sample_channels_first).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        # 转置输入以匹配PyTorch的期望格式
        batch_size = observations.shape[0]
        observations_channels_first = observations.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return self.linear(self.cnn(observations_channels_first))

# 自定义策略
class CustomCnnPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
            **kwargs
        )

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PPO运行参数")
    
    parser.add_argument("-d", "--device", type=str, default="auto", help="训练设备 (auto, cpu, cuda)")
    parser.add_argument("-a", "--action", type=str, default="train", help="运行模式 (train, test, test_random, test_keyboard)")
    parser.add_argument("-mp", "--model_path", type=str, default=None, help="模型加载或者保存路径")
    parser.add_argument("-v", "--video", action="store_true", help="是否保存视频")
    
    return parser.parse_args()

def train(args, config):
    try:
        # 确保目录存在
        os.makedirs(config.save_path, exist_ok=True)
        
        # 创建向量化环境
        env = make_vec_env(
            config.env_id, 
            n_envs=config.num_envs, 
            seed=config.seed, 
            vec_env_cls=DummyVecEnv,
            env_kwargs={
                "file_name": config.file_name,
                "no_graph": True
            }
        )
        
        # 添加 VecNormalize 包装器
        env = VecNormalize(
            env,
            norm_obs=False,  # 不标准化图像观察
            norm_reward=True,  # 标准化奖励
            clip_reward=10.0,
            gamma=config.gamma,
            epsilon=1e-8
        )
        
        # 设置保存模型的回调
        checkpoint_callback = CheckpointCallback(
            save_freq=config.save_freq,
            save_path=config.save_path,
            name_prefix=f"{config.env_id}_{config.policy}"
        )
        
        # 创建PPO模型，使用自定义策略
        model = PPO(
            CustomCnnPolicy,  # 使用自定义策略而不是默认的 CnnPolicy
            env, 
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            verbose=config.verbose,
            device=args.device,
            tensorboard_log=config.tensorboard_log
        )
        
        # 加载模型
        if args.model_path:
            model.load(args.model_path, device=args.device)
            print(f"模型权重已从 {args.model_path} 加载")
        else:
            print(f"模型不存在，开始训练...")
            
        # 训练模型
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # 保存最终模型
        final_model_path = os.path.join(config.save_path, f"{config.env_id}_{config.policy}_final")
        model.save(final_model_path)
        
        print(f"模型已保存至: {final_model_path}")
    except KeyboardInterrupt:
        print("运行中断，正在关闭环境")
    finally:
        print("环境已关闭")
        env.close()

def test(args, config, no_graph=False, save_video=False, random_action=False):
    try:
        import time 
        
        # 创建环境
        env = EpMineEnv(
            file_name=config.file_name,
            no_graph=no_graph,
            seed=config.seed,
            verbose=True,
            render_mode="human",
        )
        
        obs, _= env.reset()
        
        if save_video:
            import cv2 as cv
            video_path = "videos"
            os.makedirs(video_path, exist_ok=True)
            # 准备视频写入器
            height, width = obs.shape[:2]
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            video_writer = cv.VideoWriter(f"{video_path}/simulation_hd.mp4", fourcc, 30.0, (width, height))
        
        if not random_action:
            # 创建PPO模型
            model = PPO(
                CustomCnnPolicy,
                env, 
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                clip_range=config.clip_range,
                ent_coef=config.ent_coef,
                vf_coef=config.vf_coef,
                max_grad_norm=config.max_grad_norm,
                verbose=config.verbose,
                device=args.device
            )
            # load model
            model.load(args.model_path, device=args.device)
            
            # set eval
            model.policy.set_training_mode(False)
            
            # 测试模型
            print("开始测试模型...")
        else:
            print("开始随机动作仿真测试...")
        
        frames = []
        done = False
        step = 0
        total_reward = 0
        
        while not done:
            print(time.time())
            if random_action:
                action = env.action_space.sample()
            else:
                action, _state = model.predict(obs, deterministic=True)
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            total_reward += reward
            if save_video:
                frame = obs.copy()
                frames.append(frame)
            
            position = info["robot_position"]
            print(f"步骤: {step}, 奖励: {reward}, 位置: ({position[0]}, {position[2]})")
            print('----------------------------------------')
            step += 1
            # time.sleep(0.1)
            if step > 1000:
                break
        
        print(f"总奖励: {total_reward}")
        print(f"总步数: {step}")
        
        if save_video:
            # 将收集的帧写入视频
            for frame in frames:
                video_writer.write(frame)
            video_writer.release()
            print(f"视频已保存至: {video_path}/simulation_hd.mp4")
        
    except KeyboardInterrupt:
        print("运行中断，正在关闭环境")
        
    finally:
        print("环境已关闭")
        env.close()
        if 'video_writer' in locals() and video_writer is not None:
            video_writer.release()

def test_keyboard(args, config, no_graph=False, save_video=False):
    try:
        import time
        import cv2 as cv
        import keyboard
        
        # 创建环境
        env = EpMineEnv(
            file_name=config.file_name,
            no_graph=no_graph,
            seed=config.seed,
            time_scale=1.0,
            verbose=True,
            render_mode="human",
        )
        
        env.action_mapping = {
                0: [8.0, 0.0, 0.0],
                1: [-8.0, 0.0, 0.0],
                2: [0.0, 8.0, 0.0],
                3: [0.0, -8.0, 0.0],
                4: [0.0, 0.0, 2.0],
                5: [0.0, 0.0, -2.0],
            }
        time.sleep(1)
        
        # 键位映射字典
        key_mapping = {
            'w': 2,  # 向前
            's': 3,  # 向后
            'a': 1,  # 向左
            'd': 0,  # 向右
            'q': 5,  # 动作4
            'e': 4   # 动作5
        }
        
        # 显示键位说明
        print("键盘控制说明:")
        print("w: 向前 (2)")
        print("s: 向后 (3)")
        print("a: 向左 (0)")
        print("d: 向右 (1)")
        print("q: 动作4 (4)")
        print("e: 动作5 (5)")
        print("ESC: 退出")
        
        obs, _ = env.reset()
        
        if save_video:
            video_path = "videos"
            os.makedirs(video_path, exist_ok=True)
            # 准备视频写入器
            height, width = obs.shape[:2]
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            video_writer = cv.VideoWriter(f"{video_path}/keyboard_control.mp4", fourcc, 30.0, (width, height))
        
        frames = []
        done = False
        step = 0
        total_reward = 0
        
        print("准备好了，请按键控制...")
        
        # 主循环
        while not done:
            # 等待按键
            print("请按键选择动作 (w/a/s/d/q/e)，按ESC退出...")
            
            # 等待有效按键
            key = None
            while key not in list(key_mapping.keys()) + ['esc']:
                # 检测按键
                for k in list(key_mapping.keys()) + ['esc']:
                    if keyboard.is_pressed(k):
                        key = k
                        # 等待按键释放
                        while keyboard.is_pressed(k):
                            time.sleep(0.01)
                        break
                time.sleep(0.01)
            
            # 检查ESC退出
            if key == 'esc':
                print("用户终止测试")
                break
            
            # 执行动作
            action = key_mapping[key]
            print(f"按键: {key}, 执行动作: {action}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            total_reward += reward
            if save_video:
                frame = obs.copy()
                frames.append(frame)
            
            position = info["robot_position"]
            print(f"步骤: {step}, 奖励: {reward}, 位置: ({position[0]}, {position[2]})")
            print('----------------------------------------')
            step += 1
        
        print(f"总奖励: {total_reward}")
        print(f"总步数: {step}")
        
        if save_video:
            # 将收集的帧写入视频
            for frame in frames:
                video_writer.write(frame)
            video_writer.release()
            print(f"视频已保存至: {video_path}/keyboard_control.mp4")
        
    except KeyboardInterrupt:
        print("运行中断，正在关闭环境")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("环境已关闭")
        env.close()
        if 'video_writer' in locals() and video_writer is not None:
            video_writer.release()

if __name__ == "__main__":
    # 获取默认配置
    config = PPOConfig()
    
    # 解析命令行参数
    args = parse_args()

    if args.action == "train":
        train(args, config)
    elif args.action == "test":
        test(args, config, no_graph=False, save_video=args.video, random_action=False)
    elif args.action == "test_random":
        test(args, config, no_graph=False, save_video=args.video, random_action=True)
    elif args.action == "test_keyboard":
        test_keyboard(args, config, no_graph=False, save_video=args.video)
    else:
        raise ValueError(f"无效的运行模式: {args.action}")
