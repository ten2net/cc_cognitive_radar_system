import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.envs.registration import registry

# 列出所有已注册的CognitiveRadar环境
print("已注册的CognitiveRadar环境:")
for env_id in registry:
    # if 'CognitiveRadar' in env_id:
    print(f"  - {env_id}")
# 创建环境
env = gym.make('CognitiveRadar-v1')

# 创建模型并训练
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="./radar_tensorboard/"
)
model.learn(total_timesteps=10000)
model.save("radar_ppo_model")

env.close()