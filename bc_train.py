import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from policy_network import CustomCnnPolicy, ResNetPolicy, NatureCnnproPolicy
from ppo_custom import FilteredPPO

PPO_NAME = "ppo_custom"

ppo_policy = {
    'mlp': 'MlpPolicy',
    'cnn': 'CnnPolicy',
    'cnn_custom': CustomCnnPolicy,
    'resnet': ResNetPolicy,
    'cnn_pro': NatureCnnproPolicy,
    'lstm_cnn': "CnnLstmPolicy",
}

class ExpertDataset(Dataset):
    def __init__(self, expert_data_path=None, expert_data=None):
        if expert_data_path:
            with open(expert_data_path, 'rb') as f:
                self.expert_data = pickle.load(f)
        elif expert_data:
            self.expert_data = expert_data
        else:
            raise ValueError("专家数据路径或专家数据不能为空")
        # process data type
        self.process_data()
        
    def process_data(self):
        for data in self.expert_data:
            data['observation'] = torch.FloatTensor(data['observation'] / 255.0)
            if isinstance(data['action'], np.ndarray):
                if data['action'].ndim == 0:
                    data['action'] = data['action'].item()
            data['action'] = torch.tensor(data['action'], dtype=torch.long)
        
    def __len__(self):
        return len(self.expert_data)
    
    def __getitem__(self, idx):
        return self.expert_data[idx]

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # 类别权重
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_policy_model(model, dataloader, device, num_epochs, model_save_path):
    policy = model.policy
    policy.set_training_mode(True)
    
    # 设置优化器
    optimizer = optim.Adam(policy.parameters(), lr=5e-4, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    # 计算数据集中各动作的权重
    action_counts = {}
    for batch in dataloader:
        actions = batch['action']
        for action in actions:
            action_item = action.item()
            if action_item not in action_counts:
                action_counts[action_item] = 0
            action_counts[action_item] += 1
    
    # 计算权重（反比于频率）
    num_samples = sum(action_counts.values())
    num_classes = len(action_counts)
    class_weights = torch.zeros(num_classes)
    for action, count in action_counts.items():
        class_weights[action] = num_samples / (count * num_classes)
    
    class_weights = class_weights.to(device)
    print(f"类别权重: {class_weights}")
    
    # 使用焦点损失
    # criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # 使用带标签平滑的交叉熵
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_accuracy = 0
    loss_record = []
    accuracy_record = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for data in dataloader:
            # 将数据移到相应设备
            obs = data['observation'].to(device).squeeze()
            actions = data['action'].to(device).squeeze()
            
            # 获取策略输出
            features = policy.extract_features(obs)
            latent_pi, latent_vf = policy.mlp_extractor(features)
            action_logits = policy.action_net(latent_pi)
            
            # 计算监督学习损失 - 交叉熵
            loss = criterion(action_logits, actions)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            predicted_actions = torch.argmax(action_logits, dim=1)
            correct += (predicted_actions == actions).sum().item()
            total += actions.size(0)
        
        # 计算当前epoch的损失和准确率
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total
        
        # 更新学习率
        # scheduler.step(epoch_accuracy)
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        
        loss_record.append(epoch_loss)
        accuracy_record.append(epoch_accuracy)
        
        # 保存最佳模型
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model_path = os.path.join(model_save_path, "bc_best_model")
            model.save(best_model_path)
            print(f"保存最佳模型，准确率: {best_accuracy:.2f}%")
    
    # 绘制损失和准确率曲线,分为两个子图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_record, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_record, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_save_path, "bc_loss_accuracy_curve.png"))
    
    # 保存最终模型
    final_model_path = os.path.join(model_save_path, "bc_final_model")
    model.save(final_model_path)
    print(f"行为克隆训练完成! 最终模型已保存至: {final_model_path}")
    print(f"最佳模型已保存至: {os.path.join(model_save_path, 'bc_best_model')}, 准确率: {best_accuracy:.2f}%")

def train_behavior_cloning(expert_data_path, model_save_path, num_epochs, batch_size, device, config):
    # 加载专家数据
    dataset = ExpertDataset(expert_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建环境实例（仅用于获取空间信息）
    from envs.SingleAgent.mine_toy import EpMineEnv
    env = EpMineEnv(
        file_name=config.file_name,
        no_graph=True,
        only_image=True,
        only_state=False
    )
    
    # 创建PPO模型（我们只使用其策略网络部分）
    if PPO_NAME == "ppo":
        model = PPO(
            ppo_policy[config.policy],
            env,
            learning_rate=config.learning_rate,
            device=device
        )
    elif PPO_NAME == "ppo_recurrent":
        model = RecurrentPPO(
            ppo_policy[config.policy],
            env,
            learning_rate=config.learning_rate,
            device=device
        )
    elif PPO_NAME == "ppo_custom":
        model = FilteredPPO(
            ppo_policy[config.policy],
            env,
            reward_threshold=-8.0,
            learning_rate=config.learning_rate,
            device=device
        )
    train_policy_model(model, dataloader, device, num_epochs, model_save_path)
    
    # 关闭资源
    env.close()

def train_behavior_cloning_with_expert_model(
    config, expert_model_path, model_save_path, 
    load_model_path=None, num_epochs=10, batch_size=32, 
    device='auto', n_steps=1000):
    
    # 创建环境实例（仅用于获取空间信息）
    from envs.SingleAgent.mine_toy import EpMineEnv
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.env_util import make_vec_env
    
    expert_env = EpMineEnv(
        file_name=config.file_name,
        no_graph=False,
        only_image=False,
        only_state=False,
        max_episode_steps=512,
    )
    
    # expert_env = make_vec_env(
    #         config.env_id, 
    #         n_envs=1, 
    #         seed=config.seed, 
    #         vec_env_cls=DummyVecEnv,
    #         env_kwargs={
    #             "file_name": config.file_name,
    #             "no_graph": True,
    #             "only_image": False,
    #             "only_state": False,
    #             "max_episode_steps": 512
    #         }
    #     )
    
    env = EpMineEnv(
        file_name=config.file_name,
        no_graph=True,
        only_image=True,
        only_state=False,
        max_episode_steps=512,
    )
    
    # env = make_vec_env(
    #         config.env_id, 
    #         n_envs=1, 
    #         seed=config.seed, 
    #         vec_env_cls=DummyVecEnv,
    #         env_kwargs={
    #             "file_name": config.file_name,
    #             "no_graph": True,
    #             "only_image": True,
    #             "only_state": False,
    #             "max_episode_steps": 512
    #         }
    #     )
    
    # 加载专家模型
    # expert_model = PPO(
    #             'MlpPolicy',
    #             expert_env, 
    #             learning_rate=config.learning_rate,
    #             n_steps=config.n_steps,
    #             batch_size=config.batch_size,
    #             n_epochs=config.n_epochs,
    #             gamma=config.gamma,
    #             gae_lambda=config.gae_lambda,
    #             clip_range=config.clip_range,
    #             ent_coef=config.ent_coef,
    #             vf_coef=config.vf_coef,
    #             max_grad_norm=config.max_grad_norm,
    #             verbose=config.verbose,
    #             device=args.device
    #         )
    if PPO_NAME == "ppo":
        expert_model = PPO.load(expert_model_path, env=expert_env)
    elif PPO_NAME == "ppo_recurrent":
        expert_model = RecurrentPPO.load(expert_model_path, env=expert_env)
    elif PPO_NAME == "ppo_custom":
        expert_model = FilteredPPO.load(expert_model_path, env=expert_env)
    else:
        raise ValueError(f"不支持的PPO算法: {PPO_NAME}")
    
    if load_model_path:
        if PPO_NAME == "ppo":
            model = PPO.load(load_model_path, env=env)
        elif PPO_NAME == "ppo_recurrent":
            model = RecurrentPPO.load(load_model_path, env=env)
        elif PPO_NAME == "ppo_custom":
            model = FilteredPPO.load(load_model_path, env=env)
        else:
            raise ValueError(f"不支持的PPO算法: {PPO_NAME}")

    else:
        # 创建PPO模型（我们只使用其策略网络部分）
        if PPO_NAME == "ppo":
            model = PPO(
                    ppo_policy[config.policy],
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
                )    
        elif PPO_NAME == "ppo_recurrent":
            model = RecurrentPPO(
                    ppo_policy[config.policy],
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
                )  
        elif PPO_NAME == "ppo_custom":
            model = FilteredPPO(
                    ppo_policy[config.policy],
                    env, 
                    reward_threshold=-8.0,
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
                )   
    
    # 设置策略网络
    expert_policy = expert_model.policy
    expert_policy.set_training_mode(False)
    
    # 收集n_steps步专家数据
    total_steps = 0
    expert_data = []
    
    obs, _ = expert_env.reset()
    
    while total_steps < n_steps:
        episode_steps = 0
        done = False
        episode_data = []
        episode_reward = 0
        
        while not done:
            action, _states = expert_policy.predict(obs['state'])
            # print(f"type of action: {type(action)}") numpy.ndarray
            episode_data.append({
                'observation': obs['image'].copy(),
                'action': action.copy()
            })
            next_obs, reward, terminated, truncated, info = expert_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            # print(f"step: {episode_steps}, action: {action}, reward: {reward:.2f}")
            obs = next_obs.copy()
            episode_steps += 1
        
        obs, _ = expert_env.reset()
        print(f"episode step: {episode_steps}, episode reward: {episode_reward:.2f}")
        if episode_reward > 9.0:
            expert_data.extend(episode_data)
            total_steps += episode_steps
            print(f"total steps: {total_steps}")
    
    # 检查数据集的动作分布
    action_counts = {}
    for data in expert_data:
        action = data['action'].item() if isinstance(data['action'], np.ndarray) else data['action']
        if action not in action_counts:
            action_counts[action] = 0
        action_counts[action] += 1
    
    print("动作分布统计:")
    for action, count in action_counts.items():
        print(f"动作 {action}: {count} 样本 ({100*count/len(expert_data):.2f}%)")
        
    # 检查obs分布
    obs_data = np.array([data['observation'].copy() for data in expert_data])
    print(f"obs_data shape: {obs_data.shape}")
    print(f"obs_data min: {obs_data.min()}, max: {obs_data.max()}, mean: {obs_data.mean()}, std: {obs_data.std()}")
    
    # 加载专家数据
    dataset = ExpertDataset(expert_data=expert_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    train_policy_model(model, dataloader, device, num_epochs, model_save_path)
    
    # 关闭资源
    expert_env.close()
    env.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="bc_expert", help="训练模式，bc: 行为克隆，bc_expert: 行为克隆(使用专家模型)")
    parser.add_argument("--expert_data_path", type=str, default="expert_data/merged_expert_data.pkl", help="专家数据路径")
    parser.add_argument("--expert_model_path", type=str, default="models/expert_model.zip", help="专家模型路径")
    parser.add_argument("--load_model_path", type=str, default=None, help="加载模型路径")
    parser.add_argument("--model_save_path", type=str, default="expert_models", help="模型保存路径")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--num_epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--n_steps", type=int, default=2048, help="收集专家数据步数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    args = parser.parse_args()
    
    # 确保模型保存路径存在
    os.makedirs(args.model_save_path, exist_ok=True)
    
    from config import PPOConfig
    config = PPOConfig()
    
    if args.mode == "bc":
        train_behavior_cloning(
            args.expert_data_path, 
            args.model_save_path,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                device=args.device,
                config=config
        )
    elif args.mode == "bc_expert":
        train_behavior_cloning_with_expert_model(
            config=config,
            expert_model_path=args.expert_model_path,
            model_save_path=args.model_save_path,
            load_model_path=args.load_model_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            device=args.device,
            n_steps=args.n_steps
        )
