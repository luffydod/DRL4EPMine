import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.SingleAgent.mine_toy import EpMineEnv

from config import PPOConfig

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PPO运行参数")
    
    parser.add_argument("-d", "--device", type=str, default="auto", help="训练设备 (auto, cpu, cuda)")
    parser.add_argument("-a", "--action", type=str, default="train", help="运行模式 (train, test, test_render)")
    
    return parser.parse_args()

def train(args, config):
    try:
        # 确保目录存在
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)
        
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
        
        # 创建PPO模型
        model = PPO(
            config.policy, 
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
        
        # 训练模型
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=checkpoint_callback
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

def test(args, config):
    try:
        # 创建环境
        env = EpMineEnv(
            file_name=config.file_name,
            no_graph=False,
            render_mode="human"
        )
        # 创建PPO模型
        model = PPO(
            config.policy, 
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
        model.load(os.path.join(config.save_path, f"{config.env_id}_{config.policy}_final"))
        # 测试模型
        print("开始测试模型...")
        obs = env.reset()
        for _ in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, terminated, truncated, info = env.step(action)
            dones = terminated or truncated
    except KeyboardInterrupt:
        print("运行中断，正在关闭环境")
    finally:
        print("环境已关闭")
        env.close()

def test_render(args, config):
    import time 
    import cv2 as cv
    
    img_path = "images"
    os.makedirs(img_path, exist_ok=True)
    
    try:
        env = EpMineEnv(port=30001, no_graph=False, render_mode="human")
        obs = env.reset()
        done = False
        step = 0
        while not done:
            print(time.time())
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            position = info["robot_position"]
            cv.imwrite("{}/{}-({}, {}).png".format(img_path, step, position[0], position[2]), obs)
            print('----------------------------------------')
            step += 1
            if step > 1000:
                break
    except KeyboardInterrupt:
        print("运行中断，正在关闭环境")
    finally:
        print("环境已关闭")
        env.close()
        

if __name__ == "__main__":
    # 获取默认配置
    config = PPOConfig()
    
    # 解析命令行参数
    args = parse_args()

    if args.action == "train":
        train(args, config)
    elif args.action == "test":
        test(args, config)
    elif args.action == "test_render":
        test_render(args, config)
    else:
        raise ValueError(f"无效的运行模式: {args.action}")
