import numpy as np
import torch as th
from typing import Optional, Type, Dict, Any, List

from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
from gymnasium import spaces


class FilteredPPO(PPO):
    """
    带有轨迹筛选功能的PPO算法，可以过滤掉低于特定奖励阈值的完整轨迹
    """
    
    def __init__(
        self,
        policy,
        env,
        reward_threshold=0.0,  # 奖励阈值，低于此值的轨迹将被删除
        **kwargs
    ):
        super().__init__(policy=policy, env=env, **kwargs)
        self.reward_threshold = reward_threshold
        # 跟踪丢弃的轨迹数量
        self.discarded_episodes = 0
        self.total_episodes = 0
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        收集经验并填充到RolloutBuffer中，同时过滤掉低于奖励阈值的完整轨迹
        
        :param env: 训练环境
        :param callback: 每步调用的回调函数
        :param rollout_buffer: 用于填充的轨迹缓冲区
        :param n_rollout_steps: 每个环境要收集的经验步数
        :return: 如果至少收集了n_rollout_steps经验则返回True，如果回调提前终止则返回False
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # 切换到评估模式（影响batch norm/dropout）
        self.policy.set_training_mode(False)

        n_steps = 0
        # 记录已经添加到rollout_buffer的数据点数
        valid_data_points = 0
        rollout_buffer.reset()
        # 为状态依赖的探索采样新权重
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        
        # 跟踪每个环境的当前episode奖励
        episode_rewards = [0.0 for _ in range(env.num_envs)]
        # 跟踪每个环境的当前episode步数
        episode_steps = [0 for _ in range(env.num_envs)]
        
        # 存储轨迹临时数据
        temp_buffers = [[] for _ in range(env.num_envs)]

        # 继续收集数据，直到缓冲区满或回调终止
        while valid_data_points < rollout_buffer.buffer_size:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # 采样新的噪声矩阵
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # 转换为pytorch张量
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # 缩放并执行动作
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # 如果之前被压缩了，则反缩放动作以匹配环境边界
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # 否则，裁剪动作以避免越界错误
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # 提供访问本地变量
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # 如果是离散动作，则重新整形
                actions = actions.reshape(-1, 1)
                
            # 更新每个环境的当前episode奖励和步数
            for i in range(env.num_envs):
                episode_rewards[i] += rewards[i]
                episode_steps[i] += 1
                
                # 存储当前步骤数据到临时缓冲区
                temp_buffers[i].append({
                    "obs": self._last_obs[i].copy(),
                    "action": actions[i].copy(),
                    "reward": rewards[i],
                    "episode_start": self._last_episode_starts[i],
                    "value": values[i].cpu(),  # 确保是tensor
                    "log_prob": log_probs[i].cpu()  # 确保是tensor
                })

            # 处理超时
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value.item()
                    
                    # 更新临时缓冲区中的最后一个奖励
                    if temp_buffers[idx]:
                        temp_buffers[idx][-1]["reward"] = rewards[idx]

            # 处理完成的episode
            for i, done in enumerate(dones):
                if done:
                    self.total_episodes += 1
                    # 检查episode奖励是否低于阈值
                    if episode_rewards[i] < self.reward_threshold:
                        self.discarded_episodes += 1
                        # 清空该环境的临时缓冲区
                        temp_buffers[i] = []
                        print(f"丢弃环境 {i} 的轨迹，总奖励: {episode_rewards[i]:.2f} < {self.reward_threshold}，丢弃率: {self.discarded_episodes/self.total_episodes:.2%}")
                    else:
                        # 将临时缓冲区中的数据添加到rollout_buffer
                        for step_data in temp_buffers[i]:
                            if valid_data_points < rollout_buffer.buffer_size:
                                # 直接使用原始观察形状，不做重塑操作
                                rollout_buffer.add(
                                    np.array([step_data["obs"]]),  # 保持原始形状，只添加batch维度
                                    np.array([step_data["action"]]),
                                    np.array([step_data["reward"]]),
                                    np.array([step_data["episode_start"]]),
                                    step_data["value"],  # 已经是tensor
                                    step_data["log_prob"]  # 已经是tensor
                                )
                                valid_data_points += 1
                        temp_buffers[i] = []
                        
                    # 重置该环境的episode奖励和步数
                    episode_rewards[i] = 0.0
                    episode_steps[i] = 0
            
            # 保存观测和episode开始标志
            self._last_obs = new_obs
            self._last_episode_starts = dones
            
            # 防止无限循环（如果所有轨迹都被丢弃）
            if n_steps > n_rollout_steps * 5:  # 允许尝试5倍于原始步数
                print(f"警告：已收集 {n_steps} 步但只有 {valid_data_points}/{rollout_buffer.buffer_size} 有效数据点。")
                print(f"可能原因：奖励阈值 {self.reward_threshold} 设置过高，导致大量轨迹被丢弃。")
                # 如果收集了至少50%的需要数据，就继续
                if valid_data_points >= rollout_buffer.buffer_size * 0.5:
                    # 强制标记缓冲区为满
                    rollout_buffer.full = True
                    break
                else:
                    # 终止训练
                    return False
        
        # 确保缓冲区标记为已满
        rollout_buffer.full = True
        
        with th.no_grad():
            # 计算最后一个时间步的值
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True