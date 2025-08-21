# train_sim.py
from cognitive_radar.environment.radar_env import CognitiveRadarEnv

def random_training_run(config):
    env = CognitiveRadarEnv(config)
    obs, _ = env.reset()
    
    for episode in range(3):
        total_reward = 0
        step = 0
        
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            if step % 50 == 0:
                print(f"E:{episode} S:{step} R:{total_reward:.2f} T:{info['target_count']}")
            
            if terminated or truncated:
                break
                
        print(f"Episode {episode}结束: 总奖励 {total_reward:.2f}, 步数 {step}")
        env.reset()
    
    env.close()

# 运行测试
if __name__ == "__main__":
    config = {
        "radar_type": "fmcw_77ghz",
        "state_dim": 1024,
        "max_steps": 200,
        "action_space": {
            "type": "dict",
            "dimensions": {
                "beam_control": 2,
                "waveform_params": 3,
                "gain_control": 1
            }
        }
    }
    random_training_run(config)