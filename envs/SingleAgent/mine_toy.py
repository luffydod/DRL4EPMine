from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import ActionTuple, DecisionSteps
import numpy as np
from typing import Optional
import time
import random
import cv2 as cv
import gymnasium as gym
import os
import socket
from gymnasium import spaces

def IsOpen(port, ip='127.0.0.1'):
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    result = s.connect_ex((ip,int(port)))
    if result == 0:
        print("port {} is used".format(port))
        return True
    else:
        print("port {} is not used".format(port))
        return False

TEAM_NAME = 'ControlEP?team=0'
AGENT_ID = 0
IMAGE_SIZE = 64

def warp_action(action):
    action_dict = {'{}_{}'.format(TEAM_NAME, AGENT_ID): action}
    return action_dict

class EpMineEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self,
                 file_name: str = "MineField_Windows-0510-random/drl.exe",
                 port: Optional[int] = 30001,
                 seed: int = 0,
                 work_id: int = 0,
                 time_scale: float = 20.0,
                 max_episode_steps: int = 1000,
                 only_image: bool = True,
                 only_state: bool = False,
                 no_graph: bool = True,
                 norm_image: bool = True,
                 discrete_action: bool = True,
                 verbose: bool = False,
                 log_file: str = "logs/env_logs.txt",
                 render_mode="rgb_array"):
        super().__init__()
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.discrete_action = discrete_action
        self.norm_image = norm_image
        self.verbose = verbose
        self.log_file = log_file
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = log_file
        
        engine_configuration_channel = EngineConfigurationChannel()
        if render_mode == "human":
            engine_configuration_channel.set_configuration_parameters(
                width=1280, 
                height=720,
                time_scale=time_scale,
                target_frame_rate=60,
                capture_frame_rate=60,
            )
        else:
            engine_configuration_channel.set_configuration_parameters(
                width=200, 
                height=100,
                time_scale=time_scale,
            )
        
        self._engine_Environment_channel = EnvironmentParametersChannel()
        self.env = None
        self.port = port
        self.work_id = work_id
        self.eng_conf_channel = engine_configuration_channel
        self.env_file_name = file_name
        self.sd = seed
        self.no_graph = no_graph
        self.max_episode_length = max_episode_steps
        self.step_num = 0
        self.only_image = only_image
        self.only_state = only_state
        self.last_dist = 0.0
        self.current_results = None
        self.catch_state = 0
        
        if self.only_image:
            if self.norm_image:
                self.observation_space = spaces.Box(
                    low=0, high=1, shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32
                )
            else:
                self.observation_space = spaces.Box(
                    low=0, high=255, shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
                )
        elif self.only_state:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
                'state': spaces.Box(low=-np.Inf, high=np.Inf, shape=(7,), dtype=np.float32)
            })
            
        if self.discrete_action:
            # self.action_space = spaces.MultiDiscrete([3, 3, 3])
            # self.action_mapping = {
            #     0: [10.0, 0.0, -10.0],
            #     1: [10.0, 0.0, -10.0],
            #     2: [3.0, 0.0, -3.0]
            # }
            self.action_space = spaces.Discrete(6)
            self.action_mapping = {
                0: [1.0, 0.0, 0.0],
                1: [-1.0, 0.0, 0.0],
                2: [0.0, 1.0, 0.0],
                3: [0.0, -1.0, 0.0],
                4: [0.0, 0.0, 0.15],
                5: [0.0, 0.0, -0.15],
            }
            # self.action_mapping = {
            #     0: [5.0, 0.0, 0.0],
            #     1: [5.0, 0.0, 3.0],
            #     2: [5.0, 0.0, -3.0],
            #     3: [-5.0, 0.0, 0.0],
            #     4: [-5.0, 0.0, 3.0],
            #     5: [-5.0, 0.0, -3.0],
            #     6: [0.0, 5.0, 0.0],
            #     7: [0.0, 5.0, 3.0],
            #     8: [0.0, 5.0, -3.0],
            #     9: [0.0, -5.0, 0.0],
            #     10: [0.0, -5.0, 3.0],
            #     11: [0.0, -5.0, -3.0],
            # }
        else:
            self.action_space = spaces.Box(
                low=np.array([-10.0, -10.0, -3.0], dtype=np.float32), 
                high=np.array([10.0, 10.0, 3.0], dtype=np.float32), 
                shape=(3,), 
                dtype=np.float32
            )
        
    def seed(self, sd=0):
        self.close()
        worker_id = sd
        # 如果端口被占用，则增加worker_id
        while IsOpen(self.port+worker_id):
            worker_id += 1
        self.env = UnityEnvironment(file_name=self.env_file_name,
                                    base_port=self.port,
                                    seed=sd,
                                    worker_id=worker_id,
                                    side_channels=[self._engine_Environment_channel, self.eng_conf_channel],
                                    no_graphics=self.no_graph
                                    )

    def reset(self, *, seed=None, options=None):
        # 如果提供了种子，则设置随机数生成器
        if seed is not None:
            np.random.seed(seed)
        
        # 重置环境状态
        if self.env is None:
            self.seed(self.sd)
        self.step_num = 0
        self.env.reset()
        
        # 使用decoder_results替代_get_observation
        # 获取初始决策步骤
        decision_result = {}
        for behavior_name in self.env.behavior_specs:
            decision_result[behavior_name], _ = self.env.get_steps(behavior_name)
        
        # 解码观察结果
        observation = self.decoder_results(results=decision_result)
        
        # 初始化last_dist用于计算奖励
        self.last_dist = self.get_dist_to_mine(reuslts=decision_result)
        
        # 返回观察结果和信息字典
        info = {}  # 可以包含任何额外信息
        return observation, info
    
    def get_reward(self, results):
        reward = results[TEAM_NAME].reward[AGENT_ID]
        return reward
    
    def get_dense_reward(self, results):
        final_reward = results[TEAM_NAME].reward[AGENT_ID]
        # print(f"final_reward: {final_reward}")
        current_dist = self.get_dist_to_mine(reuslts=results)
        delta_r = (self.last_dist - current_dist) 
        final_reward += delta_r
        self.last_dist = current_dist
        return final_reward
    
    def step(self, act):
        # print(f"act: {act}")
        
        if self.discrete_action:
            # 使用离散动作映射
            # action = [self.action_mapping[i][act[i]] for i in range(len(act))]
            action = self.action_mapping[int(act)]
        else:
            # 使用连续动作
            action = act
        if self.verbose:
            print(f"action: {action}")
        
        # 添加手臂角度和抓取动作
        total_action = [action[0], action[1], action[2], 10.0, 1.0]
        
        # 创建ActionTuple并向环境发送动作
        action_tuple = ActionTuple(np.array([total_action], dtype=np.float32))
        action_dict = warp_action(action=action_tuple)
        
        # 执行动作
        total_reward = 0.0
        obs = None
        done = False
        info = {}
        
        for _ in range(1):
            obs, reward, done, info = self._step(action_dict=action_dict)
            total_reward += reward
            if done:
                break
        
        self.step_num += 1
        
        terminated = done
        truncated = (self.step_num >= self.max_episode_length)
        
        return obs, total_reward, terminated, truncated, info

    def _step(self, action_dict=None) -> DecisionSteps:
        all_agents = []
        for behavior_name in self.env.behavior_specs:
            for agent_id in self.env.get_steps(behavior_name)[0].agent_id:
                key = behavior_name + "_{}".format(agent_id)
                all_agents.append(key)
                if (action_dict != None):
                    self.env.set_action_for_agent(behavior_name, agent_id,
                                                  action_dict[key])
        self.env.step()

        decision_result = dict()
        terminal_result = dict()
        for behavior_name in self.env.behavior_specs:
            decision_result[behavior_name], terminal_result[behavior_name] = self.env.get_steps(behavior_name)
        done = False
        obs = None
        info = {}
        reward = 0.0
        if len(terminal_result[TEAM_NAME]) != 0:
            done = True
            obs = self.decoder_results(results=terminal_result)
            reward = self.get_dense_reward(results=terminal_result)
            # print(f"Pre Done! terminal_reward: {reward}", end="  ")
            self.current_results = terminal_result
            # print(f"robot_position: {self.get_robot_pose(results=terminal_result)[0]}")
            robot_position = self.get_robot_pose(results=terminal_result)[0]
            # 将信息记录到日志文件而不是打印到控制台
            log_message = f"Pre Done! terminal_reward: {reward:.1f}  robot_position: {robot_position[0]:.1f}, {robot_position[2]:.1f}"
            with open(self.log_file, "w") as f:
                f.write(log_message + "\n")
        else:
            obs = self.decoder_results(results=decision_result)
            reward = self.get_dense_reward(results=decision_result)
            self.current_results = decision_result
            robot_position = self.get_robot_pose(results=decision_result)[0]
        if self.step_num > self.max_episode_length:
            done = True
        info["robot_position"] = robot_position
        
        return obs, reward, done, info

    def decoder_results(self, results):
        org_obs = results[TEAM_NAME].obs
        if self.only_image:
            img = cv.cvtColor(np.array(org_obs[0][AGENT_ID] * 255, dtype=np.uint8), cv.COLOR_RGB2BGR)
            # 调整图像大小以匹配观察空间
            img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            if self.norm_image:
                img = img.astype(np.float32) / 255.0
            return img
        
        elif self.only_state:
            return np.array(org_obs[1][AGENT_ID][:7])
        
        else:
            img = cv.cvtColor(np.array(org_obs[0][AGENT_ID] * 255, dtype=np.uint8), cv.COLOR_RGB2BGR)
            # 调整图像大小以匹配观察空间
            img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            if self.norm_image:
                img = img.astype(np.float32) / 255.0
            rotation = org_obs[1][AGENT_ID][0:4]
            position = org_obs[1][AGENT_ID][4:7]
            arm_angle = org_obs[1][AGENT_ID][7]
            catching = org_obs[1][AGENT_ID][8]
            is_catched = org_obs[1][AGENT_ID][9]
            mineral_pose = org_obs[1][AGENT_ID][10:13]
            state = org_obs[1][AGENT_ID]
            obs = {"image": img, "state": state}
            self.catch_state = catching
            return obs
    
    def get_robot_pose(self, results):
        org_obs = results[TEAM_NAME].obs
        rotation = org_obs[1][AGENT_ID][0:4]
        position = org_obs[1][AGENT_ID][4:7]
        return position, rotation
    
    def get_mine_pose(self, results):
        org_obs = results[TEAM_NAME].obs
        mineral_pose = org_obs[1][AGENT_ID][10:13]
        return mineral_pose
    
    def get_dist_to_mine(self, reuslts):
        mine_pose = self.get_mine_pose(results=reuslts)
        # print(f"mine_pose: {mine_pose}")
        robot_pose = self.get_robot_pose(results=reuslts)[0]
        dist = np.sqrt(robot_pose[0] ** 2 + robot_pose[2] ** 2)
        return dist
    
    def render(self):
        pass
    
    def close(self):
        if self.env is not None:
            self.env.close()