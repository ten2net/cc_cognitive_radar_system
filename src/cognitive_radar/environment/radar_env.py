import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any, Optional, List

from cognitive_radar.environment.simulate_radar import RadarSimulator
from cognitive_radar.scene.scenario_manager import ScenarioManager
from cognitive_radar.target.dynamic_targets import MotionModelType


class CognitiveRadarEnv(gym.Env):
    """
    Cognitive Radar Environment for Reinforcement Learning

    This environment integrates RadarSimPy for radar simulation and 
    provides a Gymnasium-compatible interface for reinforcement learning.

    Features:
    - Dynamic radar parameter control (beam direction, waveform parameters, gain)
    - Realistic radar simulation with RadarSimPy
    - Multi-objective reward function
    - Configurable action and observation spaces
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cognitive radar environment.

        Args:
            config: Configuration dictionary containing:
                - radar_type: Type of radar (e.g., 'fmcw_77ghz')
                - radar_params: Parameters for radar factory
                - state_dim: Dimension of state vector
                - action_dim: Dimension of action vector
                - max_steps: Maximum steps per episode
                - with_background: Include background clutter
                - reward_weights: Weights for reward components
        """
        super().__init__()
        self.config = config        

        # Create radar simulator and scene manager
        self.simulator = RadarSimulator(
            radar_type=config['radar_type'],
            params=config.get('radar_params', {})
        )

        # Create scenario manager for dynamic targets
        self.scenario_manager = ScenarioManager(
            start_time=0,
            time_step=config.get('time_step', 0.1)
        )

        # Schedule targets based on config
        self._schedule_targets(config.get('targets', []))

        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config['state_dim'],),
            dtype=np.float32
        )

        # Define action space
        self._setup_action_space(config)

        # Environment state
        self.current_step = 0
        self.current_time = 0
        self.max_steps = config.get('max_steps', 500)
        # Simulation time step (seconds)
        self.time_step = config.get('time_step', 0.1)

        # Reward configuration
        self.reward_weights = config.get('reward_weights', {
            'detection': 1.0,
            'power': -0.01,
            'interference': -0.5,
            'waveform': 0.1,
            'beam': 0.5
        })

        # Visualization
        self.visualizer = None
        self.render_mode = config.get('render_mode', None)

        # Episode statistics
        self.episode_reward = 0
        self.episode_detections = 0
        self.episode_power_usage = 0

        # Radar parameter limits
        self._setup_radar_limits()

        # Initialize environment
        self.reset()

    def _schedule_targets(self, targets_config: List[Dict[str, Any]]):
        """Schedule targets based on configuration"""
        for target_config in targets_config:
            self.scenario_manager.schedule_target(
                create_time=target_config.get('create_time', 0),
                model_type=target_config['model_type'],
                **target_config['params']
            )

    def _setup_radar_limits(self):
        """Setup radar parameter limits based on radar configuration"""
        radar = self.simulator.radar
        transmitter = radar.radar_prop['transmitter']
        receiver = radar.radar_prop['receiver']

        # 辅助函数：确保值是标量
        def ensure_scalar(value):
            if isinstance(value, (list, np.ndarray)):
                return value[0]  # 取第一个元素
            return value
        
        # 确保所有值都是标量
        if 'freq_start' in transmitter.waveform_prop and 'bandwidth' in transmitter.waveform_prop:
            freq_start = ensure_scalar(transmitter.waveform_prop['freq_start'])
            bandwidth = ensure_scalar(transmitter.waveform_prop['bandwidth'])
            self.frequency_range = [freq_start, freq_start + bandwidth]
        else:
            self.frequency_range = [76e9, 81e9]  # Default for automotive radar

        # Pulse width range
        if 'pulse_width' in transmitter.waveform_prop:
            pulse_width = ensure_scalar(transmitter.waveform_prop['pulse_width'])
            self.pulse_width_range = [pulse_width * 0.5, pulse_width * 2.0]
        else:
            self.pulse_width_range = [1e-6, 100e-6]  # Default range

        # PRF range
        if 'prp' in transmitter.waveform_prop:
            prp = ensure_scalar(transmitter.waveform_prop['prp'])
            self.prf_range = [1/(prp * 2), 1/(prp * 0.5)]  # PRF = 1/PRP
        else:
            self.prf_range = [1e3, 10e3]  # Default range

        # Gain range
        if 'rf_gain' in receiver.rf_prop:
            rf_gain = ensure_scalar(receiver.rf_prop['rf_gain'])
            self.gain_range = [rf_gain * 0.5, rf_gain * 2.0]
        else:
            self.gain_range = [0, 30]  # Default gain range in dB

        # Beam angle limits
        self.max_beam_angle = 60.0  # Degrees, default value

    def _setup_action_space(self, config: Dict[str, Any]):
        """Configure action space based on config"""
        action_config = config.get('action_space', {})
        if action_config.get('type', 'flat') == 'dict':
            # Dictionary action space
            self.action_space = gym.spaces.Dict({
                'beam_control': gym.spaces.Box(
                    low=-1, high=1,
                    shape=(action_config['dimensions']['beam_control'],),
                    dtype=np.float32
                ),
                'waveform_params': gym.spaces.Box(
                    low=-1, high=1,
                    shape=(action_config['dimensions']['waveform_params'],),
                    dtype=np.float32
                ),
                'gain_control': gym.spaces.Box(
                    low=-1, high=1,
                    shape=(action_config['dimensions']['gain_control'],),
                    dtype=np.float32
                )
            })
            self._action_mapping = self._map_dict_action
        else:
            # Flat action space
            total_dims = sum(action_config['dimensions'].values()) if isinstance(action_config['dimensions'], dict) \
                else action_config.get('dimensions', 6)
            self.action_space = gym.spaces.Box(
                low=-1, high=1,
                shape=(total_dims,),
                dtype=np.float32
            )
            self._action_mapping = self._map_flat_action

    def _map_dict_action(self, action: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Map dictionary action to radar parameters"""
        params = {}

        # Beam control
        params['beam_az'] = action['beam_control'][0] * self.max_beam_angle
        params['beam_el'] = action['beam_control'][1] * self.max_beam_angle

        # Waveform parameters
        freq_min, freq_max = self.frequency_range
        pw_min, pw_max = self.pulse_width_range
        prf_min, prf_max = self.prf_range

        params['frequency'] = freq_min + \
            (freq_max - freq_min) * (0.5 * (action['waveform_params'][0] + 1))
        params['pulse_width'] = pw_min + \
            (pw_max - pw_min) * (0.5 * (action['waveform_params'][1] + 1))
        params['prf'] = prf_min + (prf_max - prf_min) * \
            (0.5 * (action['waveform_params'][2] + 1))

        # Gain control
        gain_min, gain_max = self.gain_range
        params['gain'] = gain_min + \
            (gain_max - gain_min) * (0.5 * (action['gain_control'][0] + 1))

        return params

    def _map_flat_action(self, action: np.ndarray) -> Dict[str, float]:
        """将平面动作向量映射到雷达参数"""
        # 确保动作是一维数组
        flat_action = action.flatten()
        
        # 确保使用浮点数
        params = {}
        
        # 波束控制 - 转换为浮点数
        params['beam_az'] = float(flat_action[0] * self.max_beam_angle)
        params['beam_el'] = float(flat_action[1] * self.max_beam_angle)
        
        # 波形参数
        freq_min, freq_max = map(float, self.frequency_range)
        pw_min, pw_max = map(float, self.pulse_width_range)
        prf_min, prf_max = map(float, self.prf_range)
        
        # 浮点数计算
        normalized_freq = 0.5 * (float(flat_action[2]) + 1)
        params['frequency'] = float(freq_min + (freq_max - freq_min) * normalized_freq)
        
        normalized_pw = 0.5 * (float(flat_action[3]) + 1)
        params['pulse_width'] = float(pw_min + (pw_max - pw_min) * normalized_pw)
        
        normalized_prf = 0.5 * (float(flat_action[4]) + 1)
        params['prf'] = float(prf_min + (prf_max - prf_min) * normalized_prf)
        
        # 增益控制
        gain_min, gain_max = map(float, self.gain_range)
        normalized_gain = 0.5 * (float(flat_action[5]) + 1)
        params['gain'] = float(gain_min + (gain_max - gain_min) * normalized_gain)

        return params

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step in the environment.

        Args:
            action: Action to apply to the radar system

        Returns:
            observation: New state of the environment
            reward: Reward for the action
            terminated: Whether the episode has ended
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Map action to radar parameters
        radar_params = self._action_mapping(action)
        # 确保所有参数都是标量
        for key in radar_params:
            value = radar_params[key]
            if isinstance(value, (np.ndarray, list)):
                radar_params[key] = float(value[0])  # 取第一个元素转为浮点数
            else:
                radar_params[key] = float(value)  # 确保是浮点数        

            # Update radar parameters
            self.simulator.update_radar_params(radar_params)

        # Update scene (target movement)
        self.scenario_manager.update(self.current_time)

        # Get targets in standard format
        targets = self.scenario_manager.get_targets()
        
        target_1 = dict(location=(150, 0, 0), speed=(-12, 0, 0), rcs=0.5, phase=0)
        target_2 = dict(location=(95, 20, 0), speed=(-50, 0, 0), rcs=0.5, phase=0)
        target_3 = dict(location=(30, -5, 0), speed=(-22, 0, 0), rcs=5, phase=0)

        targets = [target_1, target_2]        
        # print(">>>>>>>>>>>>>>targets", targets)
        # Run radar simulation
        baseband = self.simulator.simulate(targets)

        # Process radar data to get observation
        obs_dict = self.simulator.get_observation(baseband)
        obs = np.concatenate([
            obs_dict['rd_map'].flatten(),
            obs_dict['features']
        ])

        # Update environment state
        self.current_step += 1
        self.current_time += self.time_step

        # Calculate reward
        reward = self._calculate_reward(obs_dict, radar_params, targets)
        
        self.episode_reward += reward

        # Check termination conditions
        terminated = self.current_step >= self.max_steps
        truncated = False  # Can be set for early termination conditions

        # Information dictionary
        info = {
            'step': self.current_step,
            'time': self.current_time,
            'episode_reward': self.episode_reward,
            'detections': self.episode_detections,
            'power_usage': self.episode_power_usage,
            **radar_params,
            'target_count': len(targets)
        }

        # Add target info if available
        if targets:
            info['target_positions'] = [t['location'] for t in targets]
            info['target_speeds'] = [t['speed'] for t in targets]
            info['target_rcs'] = [t['rcs'] for t in targets]

        # Render if needed
        if self.render_mode is not None:
            print(">>>>>>>>>>>>>>>>>>>>>>>>",self.render() ,self.render_mode)

        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:  # type: ignore
        """
        Reset the environment to its initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            observation: Initial state of the environment
            info: Additional information
        """
        if seed is None:
            super().reset(seed=42)
        else:
            super().reset(seed=seed)

        # Reset environment state
        self.current_step = 0
        self.current_time = 0.0
        self.episode_reward = 0
        self.episode_detections = 0
        self.episode_power_usage = 0

        # Reset radar and scene
        self.simulator.reset_radar()
        self.scenario_manager.clear_all_targets()
        self._schedule_targets(self.config.get('targets', []))

        # Update scene to initial state
        self.scenario_manager.update(0.0)

        # Get targets in standard format
        targets = self.scenario_manager.get_targets()

        # Run initial radar simulation
        baseband = self.simulator.simulate(targets)

        # Get initial observation
        obs_dict = self.simulator.get_observation(baseband)
        obs = np.concatenate([
            obs_dict['rd_map'].flatten(),
            obs_dict['features']
        ])

        # Information dictionary
        info = {
            'step': self.current_step,
            'time': self.current_time,
            'target_count': len(targets)
        }

        # Add radar parameters if available
        radar_params = self.simulator.get_current_radar_params()
        if radar_params:
            info.update(radar_params)

        # Add target info if available
        if targets:
            info['target_positions'] = [t['location'] for t in targets]
            info['target_speeds'] = [t['speed'] for t in targets]
            info['target_rcs'] = [t['rcs'] for t in targets]

        # Reset renderer if needed
        if self.render_mode is not None:
            self._init_visualizer()

        return obs, info

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Returns:
            If render_mode is 'rgb_array', returns an RGB array.
            Otherwise, renders to screen and returns None.
        """
        if self.render_mode is None:
            return None

        if self.visualizer is None:
            self._init_visualizer()

        # Get current targets
        targets = self.scenario_manager.get_targets()

        # Update visualizer with current state
        self.visualizer.update(  # type: ignore
            processed_data=self.simulator.last_obs['rd_map'],  # type: ignore
            targets=targets,
            radar_params=self.simulator.get_current_radar_params(),
            targets_in_beam=self._get_targets_in_beam(
                targets, self.simulator.get_current_radar_params())
        )

        if self.render_mode == "rgb_array":
            return self.visualizer.get_rgb_array()  # type: ignore
        else:
            self.visualizer.render()  # type: ignore
            return None

    def close(self) -> None:
        """Clean up resources."""
        if self.visualizer is not None:
            self.visualizer.close()
            self.visualizer = None

    def _init_visualizer(self) -> None:
        """Initialize visualizer based on render mode"""
        from ..utils import RadarVisualizer
        self.visualizer = RadarVisualizer(
            render_mode=self.render_mode)  # type: ignore

    def _calculate_reward(self, obs_dict: Dict, radar_params: Dict, targets: List[Dict]) -> float:
        """
        Calculate reward based on current state and action.

        Reward components:
        1. Detection reward: Reward for detecting targets
        2. Power penalty: Penalty for high power usage
        3. Interference penalty: Penalty for interference
        4. Waveform reward: Reward for optimal waveform parameters
        5. Beam reward: Reward for beam alignment with targets

        Args:
            obs_dict: Dictionary of observation data
            radar_params: Current radar parameters
            targets: List of targets in standard format

        Returns:
            Total reward value
        """
        weights = self.reward_weights

        # 1. Detection reward
        detection_reward =self._calculate_detection_reward(obs_dict, targets) * weights['detection']
        self.episode_detections += detection_reward

        # 2. Power penalty
        power_penalty = self._calculate_power_penalty(
            radar_params) * weights['power']
        self.episode_power_usage += -power_penalty  # Since penalty is negative

        # 3. Interference penalty
        interference_penalty = self._calculate_interference_penalty(
            obs_dict) * weights['interference']

        # 4. Waveform reward
        waveform_reward = self._calculate_waveform_reward(
            radar_params) * weights['waveform']

        # 5. Beam reward
        beam_reward = self._calculate_beam_reward(
            targets, radar_params) * weights['beam']

        total_reward = (
            detection_reward +
            power_penalty +
            interference_penalty +
            waveform_reward +
            beam_reward
        )
        
        reward_dict={
            "detection_reward":detection_reward,
            "power_penalty":power_penalty,
            "interference_penalty" :interference_penalty,
            "waveform_reward":waveform_reward,
            "beam_reward":beam_reward,
            "total_reward":total_reward
        }
        
        print(reward_dict)

        return total_reward

    def _calculate_detection_reward(self, obs_dict: Dict, targets: List[Dict]) -> float:
        """使用信号幅度计算目标检测奖励"""
        processed_data = obs_dict['rd_map']
        total_reward = 0
        
        # 获取雷达当前参数
        params = self.simulator.get_current_radar_params()
        
        # 确保range_resolution有效避免除以零
        range_res = params.get('range_resolution', 0.5)  # 默认0.5米
        if range_res <= 0:
            range_res = 0.5
        
        # 同样的多普勒分辨率处理
        doppler_res = params.get('doppler_resolution', 0.2)  # 默认0.2m/s
        if doppler_res <= 0:
            doppler_res = 0.2
        for target in targets:
            x, y, z = target['location']
            # 计算真实距离
            true_range = np.sqrt(x**2 + y**2 + z**2)
            # 计算距离单元 - 确保在有效范围内
            range_bin = int(np.clip(true_range / range_res, 0, processed_data.shape[0]-1))
            
            # 计算径向速度（目标速度在视线方向上的投影）
            radial_velocity = (
                target['speed'][0] * (x/true_range) +
                target['speed'][1] * (y/true_range) +
                target['speed'][2] * (z/true_range) if true_range > 0 else 0
            )
            
            # 计算多普勒单元 - 确保在有效范围内
            doppler_bin = int(np.clip(
                radial_velocity / doppler_res, 
                0, 
                processed_data.shape[1]-1
            ))
            
            # 获取信号幅度（关键：使用np.abs转换复数到实数）            
            signal_magnitude = np.abs(processed_data[range_bin, doppler_bin])
            signal_db = 10 * np.log10(signal_magnitude + 1e-9)  # 避免log(0)
            normalized_signal = (signal_db + 100) / 100  # 将-100dB到0dB映射到0-1            
            total_reward += normalized_signal * target.get('rcs', 1.0)   
        return float(total_reward)  # 确保返回浮点数

    def _calculate_power_penalty(self, radar_params: Dict) -> float:
        """Calculate penalty for power usage"""
        # Penalize high gain and high transmit power
        gain_penalty = -radar_params.get('gain', 0)
        # Estimate transmit power (simplified)
        tx_power = radar_params.get('gain', 0) * 0.1  # Simplified model
        power_penalty = -tx_power * 0.01
        return gain_penalty + power_penalty

    def _calculate_interference_penalty(self, obs_dict: Dict) -> float:
        """计算干扰惩罚（分贝尺度+裁剪）"""
        processed_data = obs_dict['rd_map']
        
        # 获取幅度数据
        magnitude_data = np.abs(processed_data)
        
        # 转换为分贝尺度（避免log(0)）
        db_data = 10 * np.log10(magnitude_data + 1e-9)
        
        # 裁剪到合理范围（-100dB到0dB）
        clipped_db = np.clip(db_data, -100, 0)
        
        # 计算平均分贝值
        avg_db = np.mean(clipped_db)
        
        # 惩罚高干扰水平
        # 分贝值越高（越接近0），干扰越大，惩罚越大
        penalty = -avg_db * 0.01  # 缩放因子
        
        # 确保惩罚值在合理范围
        return float(np.clip(penalty, -10, 0))

    def _calculate_waveform_reward(self, radar_params: Dict) -> float:
        """Calculate reward for optimal waveform parameters"""
        reward = 0
        # Reward for high bandwidth (better resolution)
        freq = radar_params.get('frequency', self.frequency_range[0])
        # Simplified bandwidth estimate
        bandwidth = radar_params.get(
            'pulse_width', self.pulse_width_range[0]) * freq * 1e6
        max_bandwidth = self.pulse_width_range[1] * \
            self.frequency_range[1] * 1e6
        
        reward += bandwidth / max_bandwidth

        # Penalty for high PRF (increased processing load)
        prf = radar_params.get('prf', self.prf_range[0])
        max_prf = self.prf_range[1]
        reward -= 0.1 * (prf / max_prf)

        # Penalty for long pulse width (reduced time resolution)
        pulse_width = radar_params.get(
            'pulse_width', self.pulse_width_range[0])
        max_pw = self.pulse_width_range[1]
        reward -= 0.1 * (pulse_width / max_pw)
        return float(reward)

    def _calculate_beam_reward(self, targets: List[Dict], radar_params: Dict) -> float:
        """Calculate reward for beam alignment with targets"""
        # Reward for each target in the beam
        targets_in_beam = self._get_targets_in_beam(targets, radar_params)
        return len(targets_in_beam) * 2.0

    def _get_targets_in_beam(self, targets: List[Dict], radar_params: Dict) -> list:
        """Get targets currently within the radar beam"""
        targets_in_beam = []
        beam_az = radar_params.get('beam_az', 0)
        beam_el = radar_params.get('beam_el', 0)
        beam_width = 10.0  # Degrees, default beam width

        for target in targets:
            # Extract target position
            x, y, z = target['location']

            # Calculate target angles relative to radar
            target_az = np.degrees(np.arctan2(y, x))
            target_el = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))

            # Check if target is within beam
            az_diff = abs(target_az - beam_az)
            el_diff = abs(target_el - beam_el)

            if az_diff < beam_width/2 and el_diff < beam_width/2:
                targets_in_beam.append(target)

        return targets_in_beam

    def randomize_environment(self) -> None:
        """Randomize environment parameters for domain randomization"""
        # Randomize scene
        self.scenario_manager.clear_all_targets()
        self._schedule_random_targets()

        # Randomize radar parameters
        self.simulator.randomize_radar()

        # Update radar limits
        self._setup_radar_limits()

    def _schedule_random_targets(self):
        """Schedule random targets for domain randomization"""
        # Number of targets
        num_targets = np.random.randint(1, 5)

        # Schedule targets
        for i in range(num_targets):
            model_type = np.random.choice([
                MotionModelType.HIGH_SPEED_DRONE,
                MotionModelType.SWARM,
                MotionModelType.SINUSOIDAL
            ])  # type: ignore

            # Random parameters
            params = {
                'create_time': 0,
                'x_center': np.random.uniform(-100, 100),
                'y_center': np.random.uniform(-100, 100),
                'z_center': np.random.uniform(10, 100),
                'rcs': np.random.uniform(0.1, 5.0),
                'id_prefix': f"target_{i}"
            }

            if model_type == MotionModelType.HIGH_SPEED_DRONE:
                params.update({
                    'start_position': (np.random.uniform(-100, 100), np.random.uniform(-100, 100), np.random.uniform(10, 100)),
                    'end_position': (np.random.uniform(100, 500), np.random.uniform(100, 500), np.random.uniform(50, 150)),
                    'cruise_speed': np.random.uniform(30, 100),
                    'end_time': np.random.uniform(20, 60)
                })
            elif model_type == MotionModelType.SWARM:
                params.update({
                    'num_targets': np.random.randint(3, 10),
                    'area_size': np.random.uniform(20, 100)
                })

            self.scenario_manager.schedule_target(
                create_time=0,
                model_type=model_type,
                **params
            )


def main():
    import numpy as np
    from cognitive_radar.environment.radar_env import CognitiveRadarEnv

    # 配置环境
    config = {
        "radar_type": "PD-LS01",
        "state_dim": 1024,       # 示例值
        "max_steps": 500,
        "render_mode": "human",  # 可选
        "action_space": {
            "type": "dict",
            "dimensions": {
                "beam_control": 2,
                "waveform_params": 3,
                "gain_control": 1
            }
        },
        # "targets": [
        #     {
        #         "model_type": "high_speed_drone",
        #         "params": {
        #             "start_position": [50, 50, 10],
        #             "end_position": [200, 200, 20],
        #             "cruise_speed": 30,
        #             "rcs": 1.0
        #         }
        #     },
        #     {
        #         "model_type": "swarm",
        #         "params": {
        #             "num_targets": 5,
        #             "x_center": -30,
        #             "y_center": -40,
        #             "area_size": 20
        #         }
        #     }
        # ]
    }

    # 创建环境
    env = CognitiveRadarEnv(config)

    # 重置环境
    obs, info = env.reset()
    print("初始观测维度:", obs.shape)
    print("初始信息:", info.keys())

    # 随机策略测试
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"奖励: {reward:.4f}, 终止: {terminated}, 截断: {truncated}")
        print(f"雷达参数: {info['beam_az']:.1f}°, {info['prf']:.1f}Hz")

        if terminated or truncated:
            break

    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()
