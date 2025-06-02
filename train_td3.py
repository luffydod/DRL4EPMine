import os
import argparse

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.SingleAgent.mine_toy import EpMineEnv

from config import PPOConfig

from policy_network import CustomCnnPolicy, ResNetPolicy, NatureCnnproPolicy

td3_policy = {
    "mlp": "MlpPolicy",
    "cnn": "CnnPolicy",
    "cnn_custom": CustomCnnPolicy,
    "resnet": ResNetPolicy,
    "cnn_pro": NatureCnnproPolicy,
}


class TD3Config(PPOConfig):
    """TD3算法的配置，继承自PPOConfig，但覆盖一些特定于TD3的参数"""

    # TD3特有的超参数
    buffer_size: int = 100000  # 经验回放缓冲区大小
    learning_starts: int = 100  # 开始学习前需要的随机动作步数
    batch_size: int = 100  # TD3默认的批次大小
    tau: float = 0.005  # 软更新系数
    gamma: float = 0.99  # 折扣因子
    policy_delay: int = 2  # 策略更新延迟
    target_policy_noise: float = 0.2  # 目标动作噪声
    target_noise_clip: float = 0.5  # 目标动作噪声裁剪
    train_freq: int = 1  # 训练频率，默认为1
    gradient_steps: int = 1  # 每次更新的梯度步数

    # 继承其他参数但调整一些值
    learning_rate: float = 1e-3  # TD3默认学习率
    policy: str = "cnn"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="TD3运行参数")

    parser.add_argument(
        "-d", "--device", type=str, default="auto", help="训练设备 (auto, cpu, cuda)"
    )
    parser.add_argument(
        "-a",
        "--action",
        type=str,
        default="train",
        help="运行模式 (train, test, test_random, test_keyboard, collect_expert_data)",
    )
    parser.add_argument(
        "-mp", "--model_path", type=str, default=None, help="模型加载或者保存路径"
    )
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
                "no_graph": False,
                "only_image": config.only_image,
                "only_state": config.only_state,
                "discrete_action": False,  # TD3需要连续动作空间
                "max_episode_steps": 512,
            },
        )

        # 设置保存模型的回调
        checkpoint_callback = CheckpointCallback(
            save_freq=config.save_freq,
            save_path=config.save_path,
            name_prefix=f"{config.env_id}_{config.policy}",
        )

        # 创建TD3模型
        model = TD3(
            td3_policy[config.policy],
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            tau=config.tau,
            gamma=config.gamma,
            policy_delay=config.policy_delay,
            target_policy_noise=config.target_policy_noise,
            target_noise_clip=config.target_noise_clip,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            verbose=config.verbose,
            device=args.device,
            tensorboard_log=config.tensorboard_log,
        )
        print(f"TD3模型参数: {model.policy}")

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
            progress_bar=True,
        )

        # 保存最终模型
        final_model_path = os.path.join(
            config.save_path, f"{config.env_id}_{config.policy}_final"
        )
        model.save(final_model_path)

        print(f"模型已保存至: {final_model_path}")
    except KeyboardInterrupt:
        print("运行中断，正在关闭环境")
    finally:
        print("环境已关闭")
        env.close()


def test(
    args, config, test_episode=10, no_graph=False, save_video=False, random_action=False
):
    try:
        import time
        import numpy as np

        # 创建环境
        env = EpMineEnv(
            file_name=config.file_name,
            no_graph=no_graph,
            seed=config.seed,
            verbose=False,
            render_mode="human",
            only_image=config.only_image,
            only_state=config.only_state,
            discrete_action=False,  # TD3需要连续动作空间
            time_scale=config.time_scale,
        )

        obs, _ = env.reset()

        if save_video:
            import cv2 as cv

            video_path = "videos"
            os.makedirs(video_path, exist_ok=True)
            # 准备视频写入器
            height, width = obs.shape[:2]
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            video_writer = cv.VideoWriter(
                f"{video_path}/simulation_td3_hd.mp4", fourcc, 30.0, (width, height)
            )

        if not random_action:
            # 加载TD3模型
            model = TD3.load(args.model_path, env=env)

            # 设置为评估模式
            model.policy.set_training_mode(False)

            # 测试模型
            print("开始测试模型...")
        else:
            print("开始随机动作仿真测试...")

        frames = []
        done = False
        step = 0
        total_reward = 0
        success_count = 0
        success_episode_length = []
        for i in range(test_episode):
            obs_list = []

            done = False
            step = 0
            total_reward = 0
            obs, _ = env.reset()
            while not done:
                obs_list.append(obs.copy())

                if random_action:
                    action = env.action_space.sample()
                else:
                    action, _state = model.predict(obs, deterministic=True)

                next_obs, reward, terminated, truncated, info = env.step(action)
                obs = next_obs.copy()

                done = terminated or truncated
                total_reward += reward
                if save_video:
                    frame = next_obs.copy()
                    frames.append(frame)

                position = info["robot_position"]
                # print(f"步骤: {step}, 奖励: {reward}, 位置: ({position[0]}, {position[2]})")
                step += 1

            # 输出obs的分布信息
            obs_data = np.array(obs_list, dtype=np.float32)
            print(f"obs_data shape: {obs_data.shape}")
            print(
                f"obs_data min: {obs_data.min()}, max: {obs_data.max()}, mean: {obs_data.mean()}, std: {obs_data.std()}"
            )

            if total_reward > 9.0:
                success_count += 1
                success_episode_length.append(step)
            print(f"episode-{i}: 总奖励: {total_reward} 总步数: {step}")

        print(
            f"成功次数: {success_count}, 成功率: {((success_count / test_episode) * 100):.2f}%"
        )
        if success_count > 0:
            print(f"成功回合平均步数: {np.mean(success_episode_length):.1f}")

        if save_video:
            # 将收集的帧写入视频
            for frame in frames:
                video_writer.write(frame)
            video_writer.release()
            print(f"视频已保存至: {video_path}/simulation_td3_hd.mp4")

    except KeyboardInterrupt:
        print("运行中断，正在关闭环境")

    finally:
        print("环境已关闭")
        env.close()
        if "video_writer" in locals() and video_writer is not None:
            video_writer.release()


def test_keyboard(config, no_graph=False, save_video=False):
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
            discrete_action=False,  # 连续动作空间
        )
        time.sleep(1)

        # 键位映射字典 - 对于连续动作，使用预定义的连续动作值
        key_mapping = {
            "w": [0.0, 8.0, 0.0],  # 向前
            "s": [0.0, -8.0, 0.0],  # 向后
            "a": [-8.0, 0.0, 0.0],  # 向左
            "d": [8.0, 0.0, 0.0],  # 向右
            "q": [0.0, 0.0, -2.0],  # 左转
            "e": [0.0, 0.0, 2.0],  # 右转
            " ": [0.0, 0.0, 0.0],  # 不动
        }

        # 显示键位说明
        print("键盘控制说明:")
        print("w: 向前 [0.0, 8.0, 0.0]")
        print("s: 向后 [0.0, -8.0, 0.0]")
        print("a: 向左 [-8.0, 0.0, 0.0]")
        print("d: 向右 [8.0, 0.0, 0.0]")
        print("q: 左转 [0.0, 0.0, -2.0]")
        print("e: 右转 [0.0, 0.0, 2.0]")
        print("space: 不动 [0.0, 0.0, 0.0]")
        print("ESC: 退出")

        obs, _ = env.reset()

        if save_video:
            video_path = "videos"
            os.makedirs(video_path, exist_ok=True)
            # 准备视频写入器
            height, width = obs.shape[:2]
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            video_writer = cv.VideoWriter(
                f"{video_path}/td3_keyboard_control.mp4", fourcc, 30.0, (width, height)
            )

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
            while key not in list(key_mapping.keys()) + ["esc", "r"]:
                # 检测按键
                for k in list(key_mapping.keys()) + ["esc", "r"]:
                    if keyboard.is_pressed(k):
                        key = k
                        # 等待按键释放
                        while keyboard.is_pressed(k):
                            time.sleep(0.01)
                        break
                time.sleep(0.01)

            # 检查ESC退出
            if key == "esc":
                print("用户终止测试")
                break
            # 检查r重置环境
            if key == "r":
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
            print("----------------------------------------")
            step += 1

        print(f"总奖励: {total_reward}")
        print(f"总步数: {step}")

        if save_video:
            # 将收集的帧写入视频
            for frame in frames:
                video_writer.write(frame)
            video_writer.release()
            print(f"视频已保存至: {video_path}/td3_keyboard_control.mp4")

    except KeyboardInterrupt:
        print("运行中断，正在关闭环境")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print("环境已关闭")
        env.close()
        if "video_writer" in locals() and video_writer is not None:
            video_writer.release()


def collect_expert_data(config, no_graph=False):
    try:
        import time
        import keyboard

        # 创建环境
        env = EpMineEnv(
            file_name=config.file_name,
            no_graph=no_graph,
            seed=config.seed,
            time_scale=1.0,
            verbose=True,
            render_mode="human",
            discrete_action=False,  # 连续动作空间
        )
        time.sleep(1)

        # 键位映射字典 - 对于连续动作，使用预定义的连续动作值
        key_mapping = {
            "w": [0.0, 8.0, 0.0],  # 向前
            "s": [0.0, -8.0, 0.0],  # 向后
            "a": [-8.0, 0.0, 0.0],  # 向左
            "d": [8.0, 0.0, 0.0],  # 向右
            "q": [0.0, 0.0, -2.0],  # 左转
            "e": [0.0, 0.0, 2.0],  # 右转
            " ": [0.0, 0.0, 0.0],  # 不动
        }

        # 显示键位说明
        print("键盘控制说明:")
        print("w: 向前 [0.0, 8.0, 0.0]")
        print("s: 向后 [0.0, -8.0, 0.0]")
        print("a: 向左 [-8.0, 0.0, 0.0]")
        print("d: 向右 [8.0, 0.0, 0.0]")
        print("q: 左转 [0.0, 0.0, -2.0]")
        print("e: 右转 [0.0, 0.0, 2.0]")
        print("space: 不动 [0.0, 0.0, 0.0]")
        print("ESC: 退出")

        # 创建专家数据存储目录
        expert_data_dir = "expert_data_td3"
        os.makedirs(expert_data_dir, exist_ok=True)

        # 创建数据集存储列表
        expert_dataset = []

        obs, _ = env.reset()

        print("准备好了，请按键控制...")

        exit_signal = False
        while not exit_signal:
            done = False
            step = 0
            total_reward = 0
            episode_expert = []

            while not done:
                # 等待按键
                print("请按键选择动作 (w/a/s/d/q/e)，按ESC退出...，按r重置环境")

                # 等待有效按键
                key = None
                while key not in list(key_mapping.keys()) + ["esc", "r"]:
                    # 检测按键
                    for k in list(key_mapping.keys()) + ["esc", "r"]:
                        if keyboard.is_pressed(k):
                            key = k
                            # 等待按键释放
                            while keyboard.is_pressed(k):
                                time.sleep(0.01)
                            break
                    time.sleep(0.01)

                # 检查ESC退出
                if key == "esc":
                    print("用户终止测试")
                    exit_signal = True
                    break

                # 检查r重置环境
                if key == "r":
                    print("重置环境")
                    step = 0
                    obs, _ = env.reset()
                    break

                # 执行动作
                action = key_mapping[key]
                print(f"按键: {key}, 执行动作: {action}")

                # 保存当前观测和动作到专家数据集
                episode_expert.append(
                    {
                        "observation": obs.copy(),
                        "action": action,
                    }
                )

                obs, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated
                total_reward += reward

                position = info["robot_position"]
                print(
                    f"步骤: {step}, 奖励: {reward}, 位置: ({position[0]}, {position[2]})"
                )
                print("----------------------------------------")
                step += 1

            # 等待按键
            print("是否保存当前回合专家数据？(y/n)")

            # 等待有效按键
            key = None
            while key not in ["y", "n"]:
                # 检测按键
                for k in ["y", "n"]:
                    if keyboard.is_pressed(k):
                        key = k
                        # 等待按键释放
                        while keyboard.is_pressed(k):
                            time.sleep(0.01)
                        break
                time.sleep(0.01)
            if key == "y":
                expert_dataset.extend(episode_expert)
                print(f"当前回合专家数据已保存")
            else:
                print(f"当前回合专家数据未保存")

            print(f"总奖励: {total_reward}")
            print(f"总步数: {step}")

        # 保存专家数据
        import pickle
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        expert_data_path = os.path.join(
            expert_data_dir, f"expert_data_td3_{timestamp}.pkl"
        )

        with open(expert_data_path, "wb") as f:
            pickle.dump(expert_dataset, f)
        print(f"专家数据已保存至: {expert_data_path}")
    except KeyboardInterrupt:
        print("运行中断，正在关闭环境")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print("环境已关闭")
        env.close()


if __name__ == "__main__":
    # 获取默认配置
    config = TD3Config()

    # 解析命令行参数
    args = parse_args()

    if args.action == "train":
        train(args, config)
    elif args.action == "test":
        test(
            args,
            config,
            test_episode=10,
            no_graph=False,
            save_video=args.video,
            random_action=False,
        )
    elif args.action == "test_random":
        test(
            args,
            config,
            test_episode=10,
            no_graph=False,
            save_video=args.video,
            random_action=True,
        )
    elif args.action == "test_keyboard":
        test_keyboard(config, no_graph=False, save_video=args.video)
    elif args.action == "collect_expert_data":
        collect_expert_data(config, no_graph=False)
    else:
        raise ValueError(f"无效的运行模式: {args.action}") 