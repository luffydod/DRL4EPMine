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
        
        # 设置保存模型的回调
        checkpoint_callback = CheckpointCallback(
            save_freq=config.save_freq,
            save_path=config.save_path,
            name_prefix=f"{config.env_id}_{config.policy}"
        )
        
        # 创建PPO模型，使用自定义策略
        model = PPO(
            # CustomCnnPolicy,  # 使用自定义策略而不是默认的 CnnPolicy
            "MlpPolicy",  # 使用 MLP 策略
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

def test(args, config, no_graph=True, save_video=False, random_action=False, test_seed=None):
    try:
        import time
        
        # 使用指定的种子，如果有的话
        if test_seed is not None:
            seed = test_seed
        else:
            seed = config.seed
            
        print(f"初始化测试环境，使用种子: {seed}")
        
        # 重要：在创建环境前设置Python随机数种子
        import random
        random.seed(seed)
        np.random.seed(seed)
        
        # 创建与训练完全相同的向量化环境
        env = make_vec_env(
            config.env_id, 
            n_envs=1, 
            seed=seed,  # 使用传入的种子
            vec_env_cls=DummyVecEnv,
            env_kwargs={
                "file_name": config.file_name,
                "no_graph": True,
                "seed": seed  # 在env_kwargs中也传递种子
            }
        )
        
        print(f"环境创建完成，随机种子: {seed}")
        
        # 正确调用reset()，不传递seed参数
        obs = env.reset()
        print(f"初始观察形状: {type(obs)}, {np.shape(obs)}")
        print(f"初始观察: {obs}")
        
        if save_video:
            import cv2 as cv
            video_path = "videos"
            os.makedirs(video_path, exist_ok=True)
            # 准备视频写入器
            # 从向量环境获取实际观察值
            raw_obs = obs[0]
            if isinstance(raw_obs, dict) and "image" in raw_obs:
                frame = raw_obs["image"]
            else:
                # 获取底层环境的渲染
                frame = env.envs[0].render()
            
            height, width = frame.shape[:2]
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            video_writer = cv.VideoWriter(f"{video_path}/simulation_hd.mp4", fourcc, 30.0, (width, height))
        
        if not random_action:
            print(f"加载模型: {args.model_path}")
            # 确保使用与训练相同的策略
            model = PPO.load(
                args.model_path,
                env=env,
                device=args.device
            )
            
            # 明确设置为评估模式
            model.policy.set_training_mode(False)
            print("模型加载完成，设置为评估模式")
            
            # 检查模型的策略网络
            print(f"模型策略类型: {type(model.policy)}")
            print(f"观察空间: {env.observation_space}")
            print(f"动作空间: {env.action_space}")
            
            print("开始测试模型...")
        else:
            print("开始随机动作仿真测试...")
        
        frames = []
        done = False
        step = 0
        total_reward = 0
        
        while not done:
            if random_action:
                action = [env.action_space.sample()]
                print(f"随机动作: {action}")
            else:
                # 打印观察值统计信息，帮助诊断
                if step % 100 == 0:
                    print(f"第{step}步观察值形状: {np.shape(obs)}")
                    # 如果是数值观察，打印范围
                    if isinstance(obs, np.ndarray):
                        print(f"观察值范围: [{np.min(obs)}, {np.max(obs)}], 均值: {np.mean(obs)}")
                    
                action, _state = model.predict(obs, deterministic=True)
                if step % 100 == 0:
                    print(f"预测动作: {action}")
                
            # 执行动作
            obs, rewards, dones, info = env.step(action)
            
            reward = rewards[0]
            done = dones[0]
            total_reward += reward
            
            if save_video:
                # 获取底层环境的观察或渲染
                if hasattr(env.envs[0], "render"):
                    frame = env.envs[0].render()
                    frames.append(frame)
            
            # 每10步打印一次详细信息
            if step % 100 == 0:
                print(f"步骤: {step}, 奖励: {reward}")
                if "robot_position" in info[0]:
                    position = info[0]["robot_position"]
                    print(f"机器人位置: ({position[0]}, {position[2]})")
                print('----------------------------------------')
            
            if done:
                position = info[0]["robot_position"]
                print(f"机器人到达目标位置: ({position[0]}, {position[2]})")
            
            step += 1
            if step > 1000:
                break
        
        print(f"测试完成 - 总奖励: {total_reward}")
        print(f"总步数: {step}")
        
        if save_video and frames:
            # 将收集的帧写入视频
            for frame in frames:
                video_writer.write(frame)
            video_writer.release()
            print(f"视频已保存至: {video_path}/simulation_hd.mp4")
        
        # 返回测试结果
        return total_reward, step
        
    except KeyboardInterrupt:
        print("运行中断，正在关闭环境")
        return -1, 0
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return -1, 0
    finally:
        print("环境已关闭")
        env.close()
        if 'video_writer' in locals() and video_writer is not None:
            video_writer.release()

def test_multiple(args, config, num_tests=10, no_graph=True, save_video=False, random_action=False):
    """连续测试多次，统计平均性能"""
    print(f"开始连续测试{num_tests}次...")
    
    total_rewards = []
    total_steps = []
    success_count = 0
    
    # 生成不同的随机种子 - 使用时间作为基础以确保每次运行都不同
    import random
    import time
    random.seed(int(time.time()))
    base_seed = random.randint(1000, 1000000)
    test_seeds = [base_seed + i * 12345 for i in range(num_tests)]
    
    for i in range(num_tests):
        print(f"\n=== 第{i+1}/{num_tests}次测试 ===")
        
        # 使用不同的随机种子
        test_seed = test_seeds[i]
        print(f"使用随机种子: {test_seed}")
        
        reward, steps = test(args, config, no_graph=no_graph, 
                            save_video=(save_video and i==0), 
                            random_action=random_action,
                            test_seed=test_seed)
        
        if reward >= -1:  # 有效测试
            total_rewards.append(reward)
            total_steps.append(steps)
            if reward > 10.0:  # 假设奖励>9认为是成功完成任务
                success_count += 1
    
    # 计算统计数据
    if total_rewards:
        avg_reward = sum(total_rewards) / len(total_rewards)
        avg_steps = sum(total_steps) / len(total_steps)
        success_rate = success_count / len(total_rewards) * 100
        
        print("\n=== 测试统计 ===")
        print(f"完成测试: {len(total_rewards)}/{num_tests}")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"平均步数: {avg_steps:.2f}")
        print(f"成功率: {success_rate:.2f}%")
        print(f"所有奖励: {total_rewards}")
        print(f"所有步数: {total_steps}")
    else:
        print("所有测试均失败，无法计算统计数据")

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
        from gymnasium import spaces
        env.action_space = spaces.Discrete(7)
        env.action_mapping = {
                0: [8.0, 0.0, 0.0],
                1: [-8.0, 0.0, 0.0],
                2: [0.0, 8.0, 0.0],
                3: [0.0, -8.0, 0.0],
                4: [0.0, 0.0, 2.0],
                5: [0.0, 0.0, -2.0],
                6: [0.0, 0.0, 0.0],
            }
        time.sleep(1)
        
        # 键位映射字典
        key_mapping = {
            'w': 2,  # 向前
            's': 3,  # 向后
            'a': 1,  # 向左
            'd': 0,  # 向右
            'q': 5,  # 动作4
            'e': 4,   # 动作5,
            ' ': 6,   # 动作6
        }
        
        # 显示键位说明
        print("键盘控制说明:")
        print("w: 向前 (2)")
        print("s: 向后 (3)")
        print("a: 向左 (0)")
        print("d: 向右 (1)")
        print("q: 左转4 (4)")
        print("e: 右转5 (5)")
        print("space: 不动6 (6)")
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
            print("请按键选择动作 (w/a/s/d/q/e)，按ESC退出...，按r重置环境")
            
            # 等待有效按键
            key = None
            while key not in list(key_mapping.keys()) + ['esc', 'r']:
                # 检测按键
                for k in list(key_mapping.keys()) + ['esc', 'r']:
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
            # 检查r重置环境
            if key == 'r':
                print("重置环境")
                step = 0
                obs, _ = env.reset()
                continue
            
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
    elif args.action == "test_multiple":
        test_multiple(args, config, num_tests=10, no_graph=True, save_video=args.video, random_action=False)
    else:
        raise ValueError(f"无效的运行模式: {args.action}")
