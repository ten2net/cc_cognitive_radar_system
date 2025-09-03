#!/usr/bin/env python3
"""
使用 RSL-RL 训练 CognitiveRadar-v0 环境的完整示例
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from rsl_rl.env import VecEnv
from rsl_rl.algorithms import PPO
from cognitive_radar.environment.radar_env import CognitiveRadarEnv

class RadarFeatureExtractor(nn.Module):
    """处理雷达多模态输入的特征提取器"""
    
    def __init__(self, rd_map_shape: Tuple[int, int], feature_dim: int, hidden_dims: list = [256, 128]):
        super().__init__()
        
        # CNN 处理 RD 图
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        
        # 计算 CNN 输出维度
        with torch.no_grad():
            sample = torch.randn(1, 1, *rd_map_shape)
            cnn_output_dim = self.cnn(sample).shape[1]
        
        # 处理特征向量
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU()
        )
        
        # 合并特征
        self.combined_fc = nn.Sequential(
            nn.Linear(cnn_output_dim + 32, hidden_dims[0]),
            nn.ELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ELU()
        )
        
        self.output_dim = hidden_dims[-1]
    
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 处理 RD 图
        rd_map = obs_dict['rd_map'].unsqueeze(1)  # 添加通道维度 [B, C, H, W]
        cnn_features = self.cnn(rd_map)
        
        # 处理特征向量
        feature_features = self.feature_fc(obs_dict['features'])
        
        # 合并特征
        combined = torch.cat([cnn_features, feature_features], dim=1)
        return self.combined_fc(combined)

class RadarActorCritic(ActorCritic):
    """针对雷达环境定制的 Actor-Critic 网络"""
    
    def __init__(self, 
                 rd_map_shape: Tuple[int, int], 
                 feature_dim: int,
                 num_actions: int,
                 hidden_dims: list = [256, 128],
                 activation: str = "elu",
                 init_noise_std: float = 1.0):
        
        # 创建特征提取器
        self.feature_extractor = RadarFeatureExtractor(rd_map_shape, feature_dim, hidden_dims)
        features_dim = self.feature_extractor.output_dim
        
        # 初始化父类
        super().__init__(features_dim, num_actions, hidden_dims, activation=activation, init_noise_std=init_noise_std)
        
        # 存储观测值
        self.observations = None
    
    def set_observations(self, observations: Dict[str, torch.Tensor]):
        """设置当前观测值"""
        self.observations = observations
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用存储的观测值进行前向传播"""
        if self.observations is None:
            raise ValueError("Observations must be set before calling forward()")
        
        # 提取特征
        features = self.feature_extractor(self.observations)
        
        # 将特征设置为基类的观测值
        self.observations = features
        
        # 使用父类方法计算动作分布和价值
        return super().forward()

class RadarVecEnv(VecEnv):
    """雷达环境的向量化环境包装器"""
    
    def __init__(self, env_name: str, num_envs: int, config: Dict = None): # type: ignore
        self.envs = [gym.make(env_name, config=config) for _ in range(num_envs)]
        # 正确初始化基类，传递观测空间和动作空间
        super().__init__()
        self.current_observations = None
        # 确保 action_space 属性被正确设置
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
    
    def reset(self) -> TensorDict:
        observations = {}
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            for key, value in obs.items():
                if key not in observations:
                    observations[key] = np.zeros((self.num_envs, *value.shape), dtype=value.dtype)
                observations[key][i] = value
        
        # 转换为 TensorDict
        tensor_dict = {}
        for key, value in observations.items():
            tensor_dict[key] = torch.from_numpy(value).float()
        
        self.current_observations = TensorDict(tensor_dict, batch_size=[self.num_envs])
        return self.current_observations
    
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # 将张量转换为numpy数组
        actions_np = actions.cpu().numpy()
        
        observations = {}
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = {}
        
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions_np[i])
            done = terminated or truncated
            
            for key, value in obs.items():
                if key not in observations:
                    observations[key] = np.zeros((self.num_envs, *value.shape), dtype=value.dtype)
                observations[key][i] = value
            
            rewards[i] = reward
            dones[i] = done
            infos[i] = info
        
        # 转换为 TensorDict
        tensor_dict = {}
        for key, value in observations.items():
            tensor_dict[key] = torch.from_numpy(value).float()
        
        tensor_rewards = torch.from_numpy(rewards).float()
        tensor_dones = torch.from_numpy(dones).bool()
        
        self.current_observations = TensorDict(tensor_dict, batch_size=[self.num_envs])
        return self.current_observations, tensor_rewards, tensor_dones, infos
    
    def get_observations(self) -> TensorDict:
        """获取当前观测值（VecEnv 基类要求的抽象方法）"""
        if self.current_observations is None:
            raise ValueError("Observations not available. Call reset() first.")
        return self.current_observations
    
    def close(self):
        for env in self.envs:
            env.close()

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_envs(config: Dict) -> RadarVecEnv:
    """创建向量化环境"""
    env_config = {
        "radar_type": "PD-LS02",
        "max_steps": config['environment']['max_steps'],
        "action_space": {
            "type": "dict",
            "dimensions": {
                "beam_control": 2,
                "waveform_params": 3,
                "gain_control": 1
            }
        },
        "targets": [
            {
                "model_type": "HIGH_SPEED_DRONE",
                "params": {
                    "start_position": [900, 50, 50],
                    "end_position": [1000, 200, 100],
                    "cruise_speed": 30,
                    "rcs": 0.5
                }
            }
        ]
    }
    
    return RadarVecEnv(
        env_name=config['environment']['env_name'],
        num_envs=config['training']['num_envs'],
        config=env_config
    )

def create_actor_critic(env: RadarVecEnv, config: Dict) -> RadarActorCritic:
    """创建 Actor-Critic 网络"""
    # 获取观测空间信息
    sample_obs = env.reset()
    rd_map_shape = sample_obs['rd_map'].shape[1:]
    feature_dim = sample_obs['features'].shape[1]
    num_actions = env.envs[0].action_space.shape[0] # type: ignore
    
    return RadarActorCritic(
        rd_map_shape=rd_map_shape,
        feature_dim=feature_dim,
        num_actions=num_actions,
        hidden_dims=config['network']['hidden_dims'],
        activation=config['network']['activation'],
        init_noise_std=1.0
    )

def train(config_path: str = "config/radar_config.yaml"):
    """主训练函数"""
    # 加载配置
    config = load_config(config_path)
    
    # 创建环境
    env = create_envs(config)
    
    # 创建 Actor-Critic 网络
    actor_critic = create_actor_critic(env, config)
    
    # 创建 PPO 算法
    ppo = PPO(
        policy=actor_critic,
        num_learning_epochs=config['algorithm']['num_learning_epochs'],
        num_mini_batches=config['algorithm']['num_mini_batches'],
        clip_param=config['algorithm']['clip_range'],
        gamma=config['algorithm']['gamma'],
        lam=config['algorithm']['lam'],
        value_loss_coef=config['algorithm']['value_coef'],
        entropy_coef=config['algorithm']['entropy_coef'],
        learning_rate=config['algorithm']['learning_rate'],
        max_grad_norm=config['algorithm']['max_grad_norm'],
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        normalize_advantage_per_mini_batch=False
    )
    
    # 初始化存储
    sample_obs = env.reset()
    obs_shape = {key: value.shape[1:] for key, value in sample_obs.items()}
    ppo.init_storage(
        training_type="on_policy",
        num_envs=config['training']['num_envs'],
        num_transitions_per_env=config['environment']['max_steps'],
        obs=obs_shape,
        actions_shape=(env.envs[0].action_space.shape[0],) # type: ignore
    )
    
    # 创建模型保存目录
    os.makedirs("models", exist_ok=True)
    
    # 训练循环
    print("开始训练...")
    for iteration in range(config['training']['iterations']):
        # 重置环境
        obs = env.reset()
        episode_rewards = []
        
        for step in range(config['environment']['max_steps']):
            # 转换为张量
            obs_tensor = {
                key: torch.FloatTensor(value).to(ppo.device)
                for key, value in obs.items()
            }
            
            # 获取动作
            actions = ppo.act(obs_tensor)
            
            # 检查动作是否为 None
            if actions is None:
                print(f"Warning: actions is None at step {step}")
                # 使用随机动作作为后备
                actions = torch.randn((config['training']['num_envs'], env.envs[0].action_space.shape[0])).to(ppo.device) # type: ignore
            
            # 执行动作
            next_obs, rewards, dones, infos = env.step(actions.cpu().numpy()) # type: ignore
            
            # 处理环境步骤
            ppo.process_env_step(next_obs, rewards, dones, {})
            
            # 更新观测
            obs = next_obs
            
            # 记录奖励
            episode_rewards.append(np.mean(rewards))
            
            # 如果所有环境都结束，提前终止
            if np.all(dones):
                break
        
        # 计算回报
        ppo.compute_returns(obs)
        
        # 更新策略
        loss_dict = ppo.update()
        
        # 计算平均奖励
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        
        # 打印训练进度
        print(f"Iteration {iteration+1}/{config['training']['iterations']}, "
              f"Mean Reward: {mean_reward:.2f}, "
              f"Value Loss: {loss_dict['value_function']:.4f}, "
              f"Surrogate Loss: {loss_dict['surrogate']:.4f}")
        
        # 保存模型
        if (iteration + 1) % config['training']['save_interval'] == 0:
            model_path = f"models/radar_ppo_{iteration+1}.pt"
            torch.save({
                'policy_state_dict': ppo.policy.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
                'iteration': iteration,
                'mean_reward': mean_reward
            }, model_path)
            print(f"模型已保存到 {model_path}")
    
    # 关闭环境
    env.close()
    print("训练完成!")

def evaluate(model_path: str, config_path: str = "config/radar_config.yaml"):
    """评估训练好的模型"""
    # 加载配置
    config = load_config(config_path)
    
    # 创建环境
    env = gym.make(
        config['environment']['env_name'],
        config={
            "radar_type": "PD-LS02",
            "max_steps": config['environment']['max_steps'],
            "render_mode": "human",  # 启用渲染
            "action_space": {
                "type": "dict",
                "dimensions": {
                    "beam_control": 2,
                    "waveform_params": 3,
                    "gain_control": 1
                }
            },
            "targets": [
                {
                    "model_type": "HIGH_SPEED_DRONE",
                    "params": {
                        "start_position": [900, 50, 50],
                        "end_position": [1000, 200, 100],
                        "cruise_speed": 30,
                        "rcs": 0.5
                    }
                }
            ]
        }
    )
    
    # 创建 Actor-Critic 网络
    sample_obs, _ = env.reset()
    rd_map_shape = sample_obs['rd_map'].shape
    feature_dim = sample_obs['features'].shape[0]
    num_actions = env.action_space.shape[0] # type: ignore
    
    actor_critic = RadarActorCritic(
        rd_map_shape=rd_map_shape,
        feature_dim=feature_dim,
        num_actions=num_actions,
        hidden_dims=config['network']['hidden_dims'],
        activation=config['network']['activation'],
        init_noise_std=1.0
    )
    
    # 加载模型权重
    checkpoint = torch.load(model_path)
    actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
    actor_critic.eval()
    
    # 评估循环
    print("开始评估...")
    total_reward = 0
    num_episodes = 5
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(config['environment']['max_steps']):
            # 转换为张量
            obs_tensor = {
                key: torch.FloatTensor(value).unsqueeze(0)
                for key, value in obs.items()
            }
            
            # 设置观测值并获取动作
            actor_critic.set_observations(obs_tensor)
            with torch.no_grad():
                action, _, _ = actor_critic()
                action = action.squeeze(0).numpy()
            
            # 执行动作
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")
    
    env.close()
    print(f"平均奖励: {total_reward/num_episodes:.2f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练或评估 CognitiveRadar-v0 环境")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--eval", type=str, help="评估模型，指定模型路径")
    parser.add_argument("--config", type=str, default="assets/configs/radar/radar_config.yml", help="配置文件路径")
    
    args = parser.parse_args()
    
    if args.train:
        train(args.config)
    elif args.eval:
        evaluate(args.eval, args.config)
    else:
        print("请指定 --train 或 --eval 参数")