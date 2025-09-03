import gymnasium as gym
from gymnasium.envs.registration import registry
from stable_baselines3 import PPO

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class RadarFeatureExtractor(BaseFeaturesExtractor):
    """
    处理雷达多模态输入的特征提取器
    """
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        # 获取观测空间的形状
        rd_map_shape = observation_space['rd_map'].shape
        feature_dim = observation_space['features'].shape[0]
        
        # CNN 处理 RD 图
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        
        # 计算 CNN 输出维度
        with torch.no_grad():
            sample = torch.randn(1, 1, *rd_map_shape)
            cnn_output_dim = self.cnn(sample).shape[1]
        
        # 处理特征向量
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU()
        )
        
        # 合并特征
        self.combined_fc = nn.Linear(cnn_output_dim + 32, features_dim)

    def forward(self, observations):
        # 处理 RD 图
        rd_map = observations['rd_map']
        if rd_map.dim() == 3:  # 添加通道维度 [B, H, W] -> [B, 1, H, W]
            rd_map = rd_map.unsqueeze(1)
        cnn_features = self.cnn(rd_map)
        
        # 处理特征向量
        feature_vec = observations['features']
        feature_features = self.feature_fc(feature_vec)
        
        # 合并特征
        combined = torch.cat([cnn_features, feature_features], dim=1)
        return self.combined_fc(combined)

# 测试环境注册
def main():
    # 列出所有已注册的CognitiveRadar环境
    print("已注册的CognitiveRadar环境:")
    for env_id in registry:
        if 'CognitiveRadar' in env_id:
            print(f"  - {env_id}")
            
    # 测试所有环境
    for env_id in ['CognitiveRadar-v0', 'CognitiveRadar-v1']:
        try:
            env = gym.make(env_id)
            print(f"{env_id}: 创建成功")
            print(f"  观测空间: {env.observation_space}")
            print(f"  动作空间: {env.action_space}")
            env.close()
        except Exception as e:
            print(f"{env_id}: 创建失败 - {e}")
            
        # 创建环境
        env = gym.make('CognitiveRadar-v1')

        # 创建模型并训练
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs={
                "features_extractor_class": RadarFeatureExtractor,
                "features_extractor_kwargs": {"features_dim": 128},
            },
            verbose=1,
            n_steps=512,  # 减小缓冲区大小
            batch_size=32,  # 减小批次大小
            n_epochs=5,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            tensorboard_log="./radar_tensorboard/"
        )
        model.learn(total_timesteps=100)
        model.save("radar_ppo_model")

        env.close()            
if __name__ == "__main__":
    main()