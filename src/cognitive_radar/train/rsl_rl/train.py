#!/usr/bin/env python3
"""
使用 RSL-RL 训练 CognitiveRadar-v0 环境的完整示例（内存优化版）
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Union
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from rsl_rl.env import VecEnv
from rsl_rl.algorithms import PPO
import cognitive_radar.environment
import gc  # 垃圾回收
import time
import glob
from torch.utils.tensorboard import SummaryWriter

class RadarFeatureExtractor(nn.Module):
    """处理雷达多模态输入的特征提取器（极致内存优化版）"""
    
    def __init__(self, rd_map_shape: Tuple[int, int], feature_dim: int, hidden_dims: list = [64, 32]):
        super().__init__()
        
        # 简化CNN结构
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        
        # 计算CNN输出维度
        with torch.no_grad():
            sample = torch.randn(1, 1, *rd_map_shape)
            cnn_output_dim = self.cnn(sample).shape[1]
        
        # 处理特征向量
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU()
        )
        
        # 合并特征
        self.combined_fc = nn.Sequential(
            nn.Linear(cnn_output_dim + 16, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        
        # 使用更轻量的权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.output_dim = hidden_dims[-1]
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """处理多模态输入（极致内存优化）"""
        # 如果传入的是基类期望的格式（policy_obs），直接返回
        if "policy" in obs:
            return obs["policy"]
        
        # 处理 RD 图
        rd_map = obs['rd_map'].unsqueeze(1)  # 添加通道维度 [B, C, H, W]
        
        # 数值稳定性保护
        if torch.isnan(rd_map).any() or torch.isinf(rd_map).any():
            rd_map = torch.nan_to_num(rd_map, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 就地归一化节省内存
        with torch.no_grad():
            rd_map_min = rd_map.min()
            rd_map_max = rd_map.max()
            if rd_map_max - rd_map_min > 0:
                rd_map = (rd_map - rd_map_min) / (rd_map_max - rd_map_min)
        
        # 确保输入数据类型与模型参数匹配
        if next(self.cnn.parameters()).dtype != rd_map.dtype:
            rd_map = rd_map.to(next(self.cnn.parameters()).dtype)
        
        # 使用混合精度计算CNN特征
        with torch.autocast(device_type="cuda", enabled=rd_map.device.type == 'cuda'):
            cnn_features = self.cnn(rd_map)
        
        # 处理特征向量
        features = obs['features']
        
        # 数值稳定性保护
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 就地归一化节省内存
        with torch.no_grad():
            features_min = features.min()
            features_max = features.max()
            if features_max - features_min > 0:
                features = (features - features_min) / (features_max - features_min)
        
        # 确保输入数据类型与模型参数匹配
        if next(self.feature_fc.parameters()).dtype != features.dtype:
            features = features.to(next(self.feature_fc.parameters()).dtype)
        
        # 使用混合精度计算特征
        with torch.autocast(device_type="cuda", enabled=features.device.type == 'cuda'):
            feature_features = self.feature_fc(features)
        
        # 合并特征
        combined = torch.cat([cnn_features, feature_features], dim=1)
        
        # 确保输入数据类型与模型参数匹配
        if next(self.combined_fc.parameters()).dtype != combined.dtype:
            combined = combined.to(next(self.combined_fc.parameters()).dtype)
        
        # 使用混合精度计算最终输出
        with torch.autocast(device_type="cuda", enabled=combined.device.type == 'cuda'):
            output = self.combined_fc(combined)
        
        return output

class RadarActorCritic(ActorCritic):
    """针对雷达环境定制的 Actor-Critic 网络（内存优化版）"""
    
    def __init__(self, 
                 rd_map_shape: Tuple[int, int], 
                 feature_dim: int,
                 num_actions: int,
                 actor_hidden_dims: list = [128, 64],
                 critic_hidden_dims: list = [128, 64],
                 activation: str = "elu",
                 init_noise_std: float = 1.0):  
        
        # 创建特征提取器
        feature_extractor = RadarFeatureExtractor(rd_map_shape, feature_dim, actor_hidden_dims)
        features_dim = feature_extractor.output_dim
        
        # 创建符合基类要求的观测字典
        fake_obs = torch.zeros(1, features_dim)
        obs_dict = {"policy_obs": fake_obs}
        obs_groups = {"policy": ["policy_obs"], "critic": ["policy_obs"]}
        
        # 调用父类的初始化
        super().__init__(
            obs=obs_dict,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type="scalar"
        )
        
        self.feature_extractor = feature_extractor
        self.features_dim = features_dim      
        self.current_obs = None
    
    def set_observations(self, obs: Dict[str, torch.Tensor]):
        self.current_obs = obs

    def get_actor_obs(self, obs: Union[Dict[str, torch.Tensor], torch.Tensor, TensorDict]) -> torch.Tensor:
        """重写 get_actor_obs：处理字典、张量或TensorDict输入"""
        if isinstance(obs, TensorDict):
            # 将TensorDict转换为字典
            obs_dict = {key: obs[key] for key in obs.keys()} # type: ignore
            return self.get_actor_obs(obs_dict)
        elif isinstance(obs, dict):
            # 确保所有输入数据都在GPU上
            device = next(self.parameters()).device
            for key in obs:
                if isinstance(obs[key], torch.Tensor):
                    obs[key] = obs[key].to(device)
            
            if "policy_obs" in obs:
                return obs["policy_obs"]        
            return self.feature_extractor(obs)
        elif isinstance(obs, torch.Tensor):
            # 如果已经是特征向量，直接返回
            return obs
        else:
            raise ValueError(f"Expected obs to be dict, Tensor, or TensorDict, got {type(obs)}")
    
    def get_critic_obs(self, obs: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """重写 get_critic_obs：处理字典或张量输入"""
        return self.get_actor_obs(obs)
    
    def get_actor_obs_from_tensordict(self, obs: TensorDict) -> torch.Tensor:
        """从 TensorDict 获取 Actor 观测值"""
        # 将 TensorDict 转换为字典
        obs_dict = {key: obs[key] for key in obs.keys()} # type: ignore
        return self.get_actor_obs(obs_dict)
    
    def get_critic_obs_from_tensordict(self, obs: TensorDict) -> torch.Tensor:
        """从 TensorDict 获取 Critic 观测值"""
        return self.get_actor_obs_from_tensordict(obs)

class RadarVecEnv(VecEnv):
    """雷达环境的向量化环境包装器（极致内存优化版）"""
    
    def __init__(self, env_name: str, num_envs: int, config: Dict = None): # type: ignore
        self.env_name = env_name
        self.config = config
        self.num_envs = num_envs
        self.envs = []
        
        # 正确初始化基类
        super().__init__()
        self.action_space = None
        self.observation_space = None
        self.action_structure = None
        self.total_action_dim = None
        self.current_observations = None
        
        # 定义动作键顺序和维度
        self.action_keys = ['beam_control', 'waveform_params', 'gain_control']
        self.action_dims = [2, 3, 1]  # 每个动作键的维度
    
    def initialize(self):
        """延迟初始化环境以节省内存"""
        if not self.envs:
            self.envs = [gym.make(self.env_name, config=self.config) for _ in range(self.num_envs)]
            self.action_space = self.envs[0].action_space
            self.observation_space = self.envs[0].observation_space
            
            # 保存动作空间信息
            if isinstance(self.envs[0].action_space, gym.spaces.Dict):
                self.action_structure = {}
                start_idx = 0
                for key in self.action_keys:
                    size = self.action_dims[self.action_keys.index(key)]
                    self.action_structure[key] = slice(start_idx, start_idx + size)
                    start_idx += size
            else:
                self.action_structure = None
                
            # 计算总动作维度
            self.total_action_dim = sum(self.action_dims)
    
    def reset(self) -> TensorDict:
        self.initialize()
        observations = {}
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            for key, value in obs.items():
                if key not in observations:
                    observations[key] = np.zeros((self.num_envs, *value.shape), dtype=np.float32)
                observations[key][i] = value.astype(np.float32)
        
        # 转换为 TensorDict
        tensor_dict = {}
        for key, value in observations.items():
            tensor_dict[key] = torch.from_numpy(value).float()
        
        self.current_observations = TensorDict(tensor_dict, batch_size=[self.num_envs])
        return self.current_observations
    
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        self.initialize()
        # 将张量转换为numpy数组
        actions_np = actions.cpu().numpy().astype(np.float32)
        
        # 如果动作空间是字典类型，将线性动作向量转换为字典格式
        if self.action_structure:
            formatted_actions = []
            for i in range(self.num_envs):
                action_dict = {}
                for key, slc in self.action_structure.items():
                    action_dict[key] = actions_np[i, slc].flatten()
                formatted_actions.append(action_dict)
        else:
            formatted_actions = actions_np
        
        observations = {}
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = {}
        
        for i, env in enumerate(self.envs):
            try:
                obs, reward, terminated, truncated, info = env.step(formatted_actions[i])
                done = terminated or truncated
            except Exception as e:
                print(f"环境步骤出错: {e}")
                # 使用默认值继续
                obs, _ = env.reset()
                reward = -10.0  # 大惩罚
                done = True
                info = {'error': str(e)}
            
            for key, value in obs.items():
                if key not in observations:
                    observations[key] = np.zeros((self.num_envs, *value.shape), dtype=np.float32)
                observations[key][i] = value.astype(np.float32)
            
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
        if self.current_observations is None:
            raise ValueError("Observations not available. Call reset() first.")
        return self.current_observations
    
    def close(self):
        for env in self.envs:
            env.close()

def load_config(config_path: str) -> Dict:
    """加载配置文件并处理科学计数法"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 处理科学计数法
    def convert_scientific_notation(value):
        if isinstance(value, str) and 'e' in value.lower():
            try:
                return float(value)
            except ValueError:
                return value
        return value
    
    # 递归处理配置中的所有值
    def process_config(config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                process_config(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        process_config(item)
                    else:
                        value[i] = convert_scientific_notation(item)
            else:
                config_dict[key] = convert_scientific_notation(value)
    
    process_config(config)
    return config  

def create_actor_critic(env: RadarVecEnv, config: Dict) -> RadarActorCritic:
    """创建 Actor-Critic 网络（极致内存优化版本）"""
    # 获取观测空间信息
    sample_obs = env.reset()
    rd_map_shape = sample_obs['rd_map'].shape[1:]
    feature_dim = sample_obs['features'].shape[1]
    num_actions = env.total_action_dim
    
    return RadarActorCritic(
        rd_map_shape=rd_map_shape,
        feature_dim=feature_dim,
        num_actions=num_actions, # type: ignore
        actor_hidden_dims=[64, 32],  # 进一步减小隐藏层
        activation="relu",  # 使用更简单的激活函数
        init_noise_std=0.5
    )  
def create_envs(config: Dict) -> RadarVecEnv:
    """创建向量化环境（支持多环境并行）"""
    # 根据可用内存估算最大环境数量
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory
        free_mem = total_mem - torch.cuda.memory_allocated()
        # 估算每个环境的内存需求 (约500MB)
        max_envs = min(config['training']['num_envs'], int(free_mem / (500 * 1024**2)))
        print(f"GPU内存: {free_mem/1024**3:.2f} GB 可用, 最多支持 {max_envs} 个环境")
    else:
        max_envs = min(config['training']['num_envs'], 4)  # CPU上最多4个环境
    
    # 使用估算的环境数量
    num_envs = max(1, max_envs)
    
    # 简化环境配置
    env_config = {
        "radar_type": "PD-LS02",
        "max_steps": min(20, config['environment'].get('max_steps', 1000)),
        "action_space": {
            "type": "dict",
            "dimensions": {
                "beam_control": 2,
                "waveform_params": 3,
                "gain_control": 1
            }
        },
        "rd_map_resolution": (64, 64),  # 适当提高分辨率以保持性能
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
        num_envs=num_envs,
        config=env_config
    )
    
def get_next_run_dir(base_dir="runs/radar_training"):
    """获取下一个可用的运行目录"""
    # 确保基础目录存在
    os.makedirs(base_dir, exist_ok=True)
    
    # 查找所有现有运行目录
    existing_runs = glob.glob(os.path.join(base_dir, "radar_training*"))
    
    # 提取现有编号
    run_numbers = []
    for run in existing_runs:
        try:
            num = int(run.split("_")[-1])
            run_numbers.append(num)
        except ValueError:
            continue
    
    # 确定下一个编号
    next_number = max(run_numbers) + 1 if run_numbers else 1
    
    # 创建新目录
    new_dir = os.path.join(base_dir, f"radar_training_{next_number}")
    os.makedirs(new_dir, exist_ok=True)
    
    return new_dir    

def train(config_path: str = "config/radar_config.yaml"):
    """主训练函数（多环境并行版）"""
    # 加载配置
    config = load_config(config_path)
    
    # 减少训练迭代次数
    original_iterations = config['training']['iterations']
    config['training']['iterations'] = min(200, original_iterations)
    
    print(f"配置: 训练迭代数={original_iterations} -> 实际使用={config['training']['iterations']}")
    
    # 创建 TensorBoard 记录器
    # 创建唯一的日志目录
    log_dir = get_next_run_dir("rl_logs")
    writer = SummaryWriter(log_dir=log_dir)
        
    # 创建环境
    env = create_envs(config)
    num_envs = env.num_envs
    print(f"使用 {num_envs} 个并行环境")
    
    # 创建 Actor-Critic 网络
    actor_critic = create_actor_critic(env, config)
    
    # 创建 PPO 算法
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 设置PPO参数
    ppo = PPO(
        policy=actor_critic,
        num_learning_epochs=config['algorithm'].get('num_learning_epochs', 5),
        num_mini_batches=config['algorithm'].get('num_mini_batches', 4),
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
        device=device,
        normalize_advantage_per_mini_batch=False
    )
    
    # 初始化存储 - 确保容量足够大
    max_steps = min(5, config['environment'].get('max_steps', 1000))
    obs_example = torch.zeros(num_envs, actor_critic.features_dim, device=device)
    
    # 正确初始化存储，包含值函数存储
    ppo.init_storage(
        training_type="rl",
        num_envs=num_envs,
        num_transitions_per_env=max_steps + 10,  # 增加容量防止溢出
        obs={"policy": obs_example},  # 键名改为"policy"
        actions_shape=(env.total_action_dim,)
    ) 
    
    # 创建模型保存目录
    os.makedirs("models", exist_ok=True)
    
    # 训练循环（多环境并行）
    print("开始训练...")
    
    # 设置梯度检查点
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    
    # 记录训练开始时间
    start_time = time.time()    
    
    for iteration in range(config['training']['iterations']):
        # 重置环境
        obs = env.reset()
        # 初始化跟踪变量
        env_rewards = np.zeros(env.num_envs)  # 每个环境的累积奖励
        env_steps = np.zeros(env.num_envs, dtype=int)  # 每个环境的步数计数器
        completed_episodes = []  # 完成的回合列表
        total_iteration_reward = 0  # 迭代总奖励
        
        # 记录初始内存使用
        if device == 'cuda':
            print(f"Iter {iteration}: 初始内存: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        for step in range(max_steps):
            # 使用上下文管理器限制计算图范围
            with torch.autocast(device_type="cuda" if device == 'cuda' else "cpu", enabled=device == 'cuda'):            
                # 转换为张量并移动到设备
                obs_tensor = {
                    key: value.to(device)
                    for key, value in obs.items()
                }          
        
                # 设置当前观测
                actor_critic.set_observations(obs_tensor)
                
                # 使用混合精度获取特征向量
                with torch.no_grad():  # 禁用梯度以节省内存
                    # 确保输入数据类型一致
                    if next(actor_critic.feature_extractor.parameters()).dtype != obs_tensor['rd_map'].dtype:
                        obs_tensor['rd_map'] = obs_tensor['rd_map'].to(next(actor_critic.feature_extractor.parameters()).dtype)
                    if next(actor_critic.feature_extractor.parameters()).dtype != obs_tensor['features'].dtype:
                        obs_tensor['features'] = obs_tensor['features'].to(next(actor_critic.feature_extractor.parameters()).dtype)
                    
                    features = actor_critic.feature_extractor(obs_tensor)
            
                # 创建策略观测字典 - 键名改为"policy"
                policy_obs = {"policy": features}            
                
                # 使用混合精度获取动作
                actions_tensor = ppo.act(policy_obs)
            
            # 执行动作
            next_obs, rewards, dones, infos = env.step(actions_tensor.detach()) # type: ignore
            
            # 处理环境步骤
            ppo.process_env_step(next_obs, rewards, dones, {})
            
            # 累积奖励和步数
            env_rewards += rewards.cpu().numpy()
            env_steps += 1
            
            # 累加迭代总奖励
            total_iteration_reward += np.sum(rewards.cpu().numpy())            
            
            # 检查终止的环境
            done_indices = np.where(dones.cpu().numpy())[0]
            for idx in done_indices:
                # 记录完成的回合
                completed_episodes.append({
                    'env_id': idx,
                    'reward': env_rewards[idx],
                    'length': env_steps[idx],
                    'iteration': iteration
                })
                
                # 重置该环境的累积奖励和步数
                env_rewards[idx] = 0
                env_steps[idx] = 0            
            
            # 更新观测
            obs = next_obs
                        
            # 显式释放内存
            del obs_tensor, features, policy_obs, actions_tensor
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
                
            # 如果所有环境都结束，提前终止
            if dones.all():
                break
            
         # 计算回报 - 确保传入的是特征向量
        # 将当前观测移动到设备
        obs_device = obs.to(device)
        with torch.no_grad():
            # 获取当前观测的特征表示
            features = actor_critic.get_actor_obs_from_tensordict(obs_device)
            
            # 确保特征数据类型与critic网络权重匹配
            critic_weight_dtype = next(actor_critic.critic.parameters()).dtype
            if features.dtype != critic_weight_dtype:
                features = features.to(critic_weight_dtype)
        
        # 计算回报
        ppo.compute_returns(features)
        
        # 使用混合精度更新策略
        with torch.autocast(device_type="cuda" if device == 'cuda' else "cpu", enabled=device == 'cuda'):   
            loss_dict = ppo.update()            
            
        
        # 处理未完成的回合
        for i in range(env.num_envs):
            if env_steps[i] > 0:  # 未完成但有数据
                completed_episodes.append({
                    'env_id': i,
                    'reward': env_rewards[i],
                    'length': env_steps[i],
                    'iteration': iteration,
                    'incomplete': True
                })
        
        # 计算迭代指标
        # 1. 迭代总奖励
        writer.add_scalar('Reward/Total_Reward', total_iteration_reward, iteration)
        # 初始化默认值
        mean_episode_reward = 0
        mean_episode_length = 0
        episodes_completed = 0        
        # 2. 平均回合奖励（仅计算完成的回合）
        if completed_episodes:
            # 筛选完成的回合
            complete_episodes = [ep for ep in completed_episodes if 'incomplete' not in ep]
            
            if complete_episodes:
                episode_rewards = [ep['reward'] for ep in complete_episodes]
                mean_episode_reward = np.mean(episode_rewards)
                writer.add_scalar('Reward/Mean_Episode_Reward', mean_episode_reward, iteration)
                
                # 记录回合奖励分布
                writer.add_histogram('Histogram/Episode_Reward_Distribution', np.array(episode_rewards), iteration)
            
            # 3. 平均回合长度
            episode_lengths = [ep['length'] for ep in complete_episodes]
            mean_episode_length = np.mean(episode_lengths) if episode_lengths else 0
            writer.add_scalar('Metrics/Mean_Episode_Length', mean_episode_length, iteration)
            
            # 4. 完成的回合数
            episodes_completed = len(complete_episodes)
            writer.add_scalar('Metrics/Episodes_Completed', episodes_completed, iteration)
        
        # 记录每个环境的详细数据
        for ep in completed_episodes:
            # 累积奖励
            writer.add_scalar(f'Reward/Episode_Reward/Env_{ep["env_id"]}', 
                             ep['reward'], ep['iteration'] * env.num_envs + ep['env_id'])
            
            # 回合长度
            writer.add_scalar(f'Metrics/Episode_Length/Env_{ep["env_id"]}', 
                             ep['length'], ep['iteration'] * env.num_envs + ep['env_id'])   

        # 记录训练指标
        writer.add_scalar('Train/Value_Loss', loss_dict['value_function'], iteration)
        writer.add_scalar('Train/Surrogate_Loss', loss_dict['surrogate'], iteration)
        writer.add_scalar('Train/Entropy_Loss', loss_dict['entropy'], iteration)
        for i, param_group in enumerate(ppo.optimizer.param_groups):
            writer.add_scalar(f'Train/Learning_Rate/Group_{i}', param_group['lr'], iteration)
        
        # 记录内存使用情况
        if device == 'cuda':
            writer.add_scalar('Memory/GPU_Allocated_MB', torch.cuda.memory_allocated()/1024**2, iteration)
            writer.add_scalar('Memory/GPU_Reserved_MB', torch.cuda.memory_reserved()/1024**2, iteration)
        
        # 添加参数分布直方图记录  
        if iteration % 10 == 0:
            for name, param in actor_critic.named_parameters():
                writer.add_histogram(f"Histogram/{name}", param, iteration)  
                
        # 释放内存
        del obs, obs_device
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()   
                                       
        # 打印训练进度
        print(f"Iteration {iteration+1}/{config['training']['iterations']}: "
              f"Total Reward: {total_iteration_reward:.2f}, "
              f"Mean Episode Reward: {mean_episode_reward:.2f}, "
              f"Mean Length: {mean_episode_length:.1f}, "
              f"Episodes: {episodes_completed}")        
        
        # 保存模型
        if (iteration + 1) % max(1, config['training']['save_interval']) == 0:
            model_path = f"models/radar_ppo_{iteration+1}.pt"
            torch.save({
                'policy_state_dict': ppo.policy.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
                'iteration': iteration,
                'ep_reward': total_iteration_reward
            }, model_path)
            print(f"模型已保存到 {model_path}")
    
    # 关闭环境
    env.close()
    
    # 记录总训练时间
    total_time = time.time() - start_time
    print(f"训练完成! 总耗时: {total_time/60:.2f} 分钟")
    
    # 关闭TensorBoard写入器
    writer.close()

def evaluate(model_path: str, config_path: str = "config/radar_config.yaml"):
    """评估训练好的模型（内存优化版本）"""
    # 加载配置
    config = load_config(config_path)
    
    # 创建环境（仅使用1个）
    env = gym.make(
        config['environment']['env_name'],
        config={
            "radar_type": "PD-LS02",
            "max_steps": min(200, config['environment'].get('max_steps', 1000)),
            "render_mode": "human",
            "action_space": {
                "type": "dict",
                "dimensions": {
                    "beam_control": 2,
                    "waveform_params": 3,
                    "gain_control": 1
                }
            },
            "rd_map_resolution": (64, 64),
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
        actor_hidden_dims=[64, 32],
        activation="relu",
        init_noise_std=0.5
    )
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location='cpu')
    actor_critic.load_state_dict(checkpoint['policy_state_dict'])
    actor_critic.eval()
    
    # 评估循环
    print("开始评估...")
    total_reward = 0
    num_episodes = 2  # 减少评估次数节省内存
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(config['environment']['max_steps']):
            # 转换为张量
            obs_tensor = {
                key: torch.FloatTensor(value).unsqueeze(0).to('cpu')
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
        # 释放内存
        gc.collect()
    
    env.close()
    print(f"平均奖励: {total_reward/num_episodes:.2f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练或评估 CognitiveRadar-v0 环境")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--eval", type=str, help="评估模型，指定模型路径")
    parser.add_argument("--config", type=str, default="assets/configs/radar/radar_config.yml", help="配置文件路径")
    
    args = parser.parse_args()
    print(f"使用配置文件: {args.config}")
    if args.train:
        train(args.config)
    elif args.eval:
        evaluate(args.eval, args.config)
    else:
        print("请指定 --train 或 --eval 参数")