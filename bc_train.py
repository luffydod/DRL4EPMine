import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from stable_baselines3 import PPO

class ExpertDataset(Dataset):
    def __init__(self, expert_data_path):
        with open(expert_data_path, 'rb') as f:
            self.expert_data = pickle.load(f)
        
    def __len__(self):
        return len(self.expert_data)
    
    def __getitem__(self, idx):
        data = self.expert_data[idx]
        # obs = torch.FloatTensor(data['observation']) / 255.0  # 归一化图像
        obs = torch.FloatTensor(data['observation'])
        action = torch.LongTensor([data['action']])
        return obs, action

def train_behavior_cloning(expert_data_path, model_save_path, num_epochs, batch_size, device, config):
    # 加载专家数据
    dataset = ExpertDataset(expert_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建环境实例（仅用于获取空间信息）
    from envs.SingleAgent.mine_toy import EpMineEnv
    env = EpMineEnv(
        file_name=config.file_name,
        no_graph=True,
        only_image=config.only_image,
        only_state=config.only_state
    )
    
    # 创建策略网络
    if config.policy == 'cnn':
        policy_type = 'CnnPolicy'
    elif config.policy == 'mlp':
        policy_type = 'MlpPolicy'
    elif config.policy == 'resnet':
        from policy_network import ResNetPolicy
        policy_type = ResNetPolicy
    
    # 创建PPO模型（我们只使用其策略网络部分）
    model = PPO(
        policy_type,
        env,
        learning_rate=config.learning_rate,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 设置策略网络为训练模式
    policy = model.policy
    policy.set_training_mode(True)
    
    # 设置优化器
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)
    
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for obs, actions in dataloader:
            # 将数据移到相应设备
            obs = obs.to(device)
            actions = actions.to(device).squeeze()
            
            # 直接使用策略网络的前向传播
            # 注意：这里不使用model.predict，因为那是用于推理
            distribution = policy.get_distribution(obs)
            action_logits = distribution.distribution.logits
            
            # 计算损失
            loss = criterion(action_logits, actions)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            predicted_actions = torch.argmax(action_logits, dim=1)
            correct += (predicted_actions == actions).sum().item()
            total += actions.size(0)
        
        # 计算当前epoch的损失和准确率
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        
        # 保存最佳模型
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model_path = os.path.join(model_save_path, "bc_best_model")
            model.save(best_model_path)
            print(f"保存最佳模型，准确率: {best_accuracy:.2f}%")
    
    # 保存最终模型
    final_model_path = os.path.join(model_save_path, "bc_final_model")
    model.save(final_model_path)
    print(f"行为克隆训练完成! 最终模型已保存至: {final_model_path}")
    print(f"最佳模型已保存至: {os.path.join(model_save_path, 'bc_best_model')}, 准确率: {best_accuracy:.2f}%")
    
    return model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_data_path", type=str, default="expert_data/merged_expert_data.pkl", help="专家数据路径")
    parser.add_argument("--model_save_path", type=str, default="models", help="模型保存路径")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--num_epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    args = parser.parse_args()
    
    # 确保模型保存路径存在
    os.makedirs(args.model_save_path, exist_ok=True)
    
    from config import PPOConfig
    config = PPOConfig()
    # 将命令行参数更新到配置中
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    
    train_behavior_cloning(
        args.expert_data_path, 
        args.model_save_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        device=args.device,
        config=config
    )
