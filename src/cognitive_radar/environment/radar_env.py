import gymnasium as gym
import numpy as np
import math
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

    def __init__(self, config: Dict[str, Any]={}, **kwargs):
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
        
        # 设置默认配置
        default_config = {
            "radar_type": "PD-LS02",
            "max_steps": 500,
            "time_step": 0.1,
            "action_space": {
                "type": "dict",
                "dimensions": {
                    "beam_control": 2,
                    "waveform_params": 3,
                    "gain_control": 1
                }
            },
            "reward_weights": {
                'detection': 1.0,
                'power': -0.01,
                'interference': -0.5,
                'waveform': 0.1,
                'beam': 0.5
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
        
        # 合并配置
        if config is not None:
            final_config = {**default_config, **config}
        else:
            final_config = default_config
            
        # 合并额外的关键字参数
        final_config = {**final_config, **kwargs}
        
        self.config = final_config    

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

        # 获取RD图形状和特征维度
        self._get_observation_shapes()
        
        # Define observation space as Dict (不再使用flatten)
        self.observation_space = gym.spaces.Dict({
            'rd_map': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.rd_map_shape,
                dtype=np.float32
            ),
            'features': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.features_dim,),
                dtype=np.float32
            )
        })

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
        
    def _get_observation_shapes(self):
        """获取RD图形状和特征维度"""
        # 运行一次仿真来获取形状信息
        self.simulator.reset_radar()
        self.scenario_manager.clear_all_targets()
        self._schedule_targets(self.config.get('targets', []))
        self.scenario_manager.update(0.0)
        targets = self.scenario_manager.get_targets()
        
        # 使用简单目标进行测试
        test_targets = [
            dict(location=(1000, 0, 100), speed=(-11.2, 0, 0), rcs=0.5, phase=0),
            dict(location=(2000, 0, 100), speed=(30, 0, 0), rcs=0.5, phase=0)
        ]
        
        baseband = self.simulator.simulate(test_targets)
        obs_dict = self.simulator.get_observation(baseband)
        
        self.rd_map_shape = obs_dict['rd_map'].shape
        self.features_dim = obs_dict['features'].shape[0]
        
        # 重置环境
        self.simulator.reset_radar()
        self.scenario_manager.clear_all_targets()

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
        transmitter = radar.radar_prop['transmitter'] # type: ignore
        receiver = radar.radar_prop['receiver'] # type: ignore

        # 辅助函数：确保值是标量
        def ensure_scalar(value):
            if isinstance(value, (list, np.ndarray)):
                return value[0]  # 取第一个元素
            return value
        
        # 确保所有值都是标量
        if 'start' in transmitter.waveform_prop and 'bandwidth' in transmitter.waveformprop:
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
        """Map dictionary action to radar parameters with enhanced numerical stability"""
        # 验证输入动作值
        if np.isnan(action['beam_control']).any() or np.isinf(action['beam_control']).any():
            print(f"警告: beam_control 包含 NaN 或 Inf 值! 使用零值替代")
            action['beam_control'] = np.zeros_like(action['beam_control'])
        
        if np.isnan(action['waveform_params']).any() or np.isinf(action['waveform_params']).any():
            print(f"警告: waveform_params 包含 NaN 或 Inf 值! 使用零值替代")
            action['waveform_params'] = np.zeros_like(action['waveform_params'])
        
        if np.isnan(action['gain_control']).any() or np.isinf(action['gain_control']).any():
            print(f"警告: gain_control 包含 NaN 或 Inf 值! 使用零值替代")
            action['gain_control'] = np.zeros_like(action['gain_control'])
        
        # 限制动作值在[-1,1]范围内
        beam_control = np.clip(action['beam_control'], -1.0, 1.0)
        waveform_params = np.clip(action['waveform_params'], -1.0, 1.0)
        gain_control = np.clip(action['gain_control'], -1.0, 1.0)
        
        # Beam control
        params = {}
        params['beam_az'] = beam_control[0] * self.max_beam_angle
        params['beam_el'] = beam_control[1] * self.max_beam_angle
        
        # Waveform parameters
        freq_min, freq_max = self.frequency_range
        pw_min, pw_max = self.pulse_width_range
        prf_min, prf_max = self.prf_range
        
        # 确保频率计算在有效范围内
        freq_factor = 0.5 * (waveform_params[0] + 1)
        freq_factor = np.clip(freq_factor, 0.0, 1.0)
        params['frequency'] = freq_min + (freq_max - freq_min) * freq_factor
        if params['frequency'] < freq_min or params['frequency'] > freq_max:
            # 使用更严格的限制方法
            params['frequency'] = np.clip(params['frequency'], freq_min, freq_max)        
        
        # 确保脉冲宽度计算在有效范围内
        pw_factor = 0.5 * (waveform_params[1] + 1)
        pw_factor = np.clip(pw_factor, 0.0, 1.0)
        params['pulse_width'] = pw_min + (pw_max - pw_min) * pw_factor
        
        # 确保PRF计算在有效范围内
        prf_factor = 0.5 * (waveform_params[2] + 1)
        prf_factor = np.clip(prf_factor, 0.0, 1.0)
        params['prf'] = prf_min + (prf_max - prf_min) * prf_factor
        
        # Gain control
        gain_min, gain_max = self.gain_range
        gain_factor = 0.5 * (gain_control[0] + 1)
        gain_factor = np.clip(gain_factor, 0.0, 1.0)
        params['gain'] = gain_min + (gain_max - gain_min) * gain_factor
        
        # 验证参数有效性
        if params['pulse_width'] <= 0:
            print(f"警告: 脉冲宽度无效 ({params['pulse_width']})，使用最小值 {pw_min}")
            params['pulse_width'] = pw_min
        
        if params['prf'] <= 0:
            print(f"警告: PRF无效 ({params['prf']})，使用最小值 {prf_min}")
            params['prf'] = prf_min
        
        # 确保频率在范围内
        if params['frequency'] < freq_min or params['frequency'] > freq_max:
            print(f"警告: 频率无效 ({params['frequency']})，限制在范围内 {freq_min}-{freq_max}")
            params['frequency'] = np.clip(params['frequency'], freq_min, freq_max)
        
        # 使用float32确保数值精度
        return {
            'beam_az': float(params['beam_az']),
            'beam_el': float(params['beam_el']),
            'frequency': float(params['frequency']),
            'pulse_width': float(params['pulse_width']),
            'prf': float(params['prf']),
            'gain': float(params['gain'])
        }            

    def _map_flat_action(self, action: np.ndarray) -> Dict[str, float]:
        """将平面动作向量映射到雷达参数"""
        # 确保动作是一维数组
        flat_action = action.flatten()
        
        # 确保使用浮点数
        params = {}
        
        # 波束控制 - 转换为浮点数
        # 限制波束角度范围
        max_az_angle = 60.0  # 最大方位角（度）
        max_el_angle = 45.0  # 最大俯仰角（度）
        
        params['beam_az'] = float(np.clip(action[0] * max_az_angle, -max_az_angle, max_az_angle))
        params['beam_el'] = float(np.clip(action[1] * max_el_angle, -max_el_angle, max_el_angle))
        
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

    def step(self, action: Any) -> Tuple[Dict, float, bool, bool, Dict[str, Any]]:
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
        self.simulator.update_radar(radar_params)
        self.simulator.reset_radar()
        # Update scene (target movement)
        self.scenario_manager.update(self.current_time)

        # Get targets in standard format
        targets = self.scenario_manager.get_targets()
        
        # 使用测试目标
        # target_1 = dict(location=(1000+ 100 * self.current_step , 0, 100), speed=(-110.2, 0, 0), rcs=0.5, phase=0)
        # target_2 = dict(location=(2000 - 100 * self.current_step, 0, 100), speed=(90 , 0, 0), rcs=0.5, phase=0)
        # targets = [target_1, target_2]             
        
        # Run radar simulation
        baseband = self.simulator.simulate(targets)

        # Process radar data to get observation
        obs_dict = self.simulator.get_observation(baseband)
        
        # 使用字典格式的观测值（不再flatten）
        obs = {
            'rd_map': obs_dict['rd_map'],
            'features': obs_dict['features']
        }

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
            self.render()

        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict[str, Any]]:
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
        
        # 使用字典格式的观测值（不再flatten）
        obs = {
            'rd_map': obs_dict['rd_map'].astype(np.float32),
            'features': obs_dict['features'].astype(np.float32)
        }

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
        reward_dict = {k: round(v, 4) if isinstance(v, float) else v 
                 for k, v in reward_dict.items()}
        print(reward_dict)

        return total_reward
    
    def _calculate_detection_reward(self, obs_dict: Dict, targets: List[Dict]) -> float:
        """使用峰值检测算法计算检测奖励"""
        processed_data = obs_dict['rd_map']
        total_reward = 0
        
        # 获取雷达参数
        params = self.simulator.get_current_radar_params()
        range_res = params.get('range_resolution', 0.5)
        doppler_res = params.get('doppler_resolution', 0.2)
        
        # 使用雷达模拟器的峰值检测功能
        peaks = self.simulator.find_peaks(processed_data, num_peaks=len(targets) + 2)
        
        print("="*50)
        print(f"检测到的峰值数量: {len(peaks)}")
        print(f"RD图形状: {processed_data.shape}")
        print(f"距离分辨率: {range_res:.3f}m, 多普勒分辨率: {doppler_res:.3f}m/s")
        
        # 为每个目标寻找匹配的峰值
        matched_targets = set()
        matched_peaks = set()
        
        for i, target in enumerate(targets):
            print(f"\n目标 #{i+1}:")
            x, y, z = target['location']
            print(f"位置: ({x:.1f}, {y:.1f}, {z:.1f})m")
            
            # 计算真实距离和径向速度
            true_range = np.sqrt(x**2 + y**2 + z**2)
            
            # 计算径向速度
            radial_velocity = (
                target['speed'][0] * (x/true_range) +
                target['speed'][1] * (y/true_range) +
                target['speed'][2] * (z/true_range) if true_range > 0 else 0
            )
            
            print(f"真实距离: {true_range:.1f}m, 径向速度: {radial_velocity:.1f}m/s")
            
            # 计算对应的距离和多普勒单元
            expected_range_bin = int(np.clip(true_range / range_res, 0, processed_data.shape[1]-1))
            
            # 多普勒索引计算（需要考虑多普勒域的对称性）
            max_doppler = (processed_data.shape[0] // 2) * doppler_res
            doppler_bin_float = radial_velocity / doppler_res
            expected_doppler_bin = int(np.clip(
                doppler_bin_float + processed_data.shape[0] // 2,
                0,
                processed_data.shape[0] - 1
            ))
            
            print(f"预期单元: 距离={expected_range_bin}, 多普勒={expected_doppler_bin}")
            print(f"预期物理量: 距离={true_range:.1f}m, 速度={radial_velocity:.1f}m/s")
            
            # 寻找最匹配的峰值
            best_match = None
            best_distance = float('inf')
            
            for j, peak in enumerate(peaks):
                if j in matched_peaks:
                    continue
                
                # 获取峰值的物理距离和速度
                peak_range = peak['range']
                peak_velocity = peak['velocity']
                
                # 计算物理距离差异（米和米/秒）
                range_diff_m = abs(peak_range - true_range)
                velocity_diff = abs(peak_velocity - radial_velocity)
                
                # 综合距离差异（加权）
                distance = math.sqrt(range_diff_m**2 + (velocity_diff * 10)**2)  # 速度差异权重较大
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = (peak, j)
            
            # 检查是否找到匹配的峰值
            match_threshold = 10.0  # 10米的匹配阈值（更严格）
            if best_match and best_distance < match_threshold:
                peak, peak_idx = best_match
                matched_targets.add(i)
                matched_peaks.add(peak_idx)
                
                # 计算信号强度
                signal_db = peak['intensity']
                print(f"匹配峰值: 距离={peak['range']:.1f}m, 速度={peak['velocity']:.1f}m/s")
                print(f"信号强度: {signal_db:.1f}dB, SNR: {peak['snr']:.1f}dB")
                
                # 计算距离差异
                range_error = abs(peak['range'] - true_range)
                velocity_error = abs(peak['velocity'] - radial_velocity)
                
                print(f"距离误差: {range_error:.1f}m, 速度误差: {velocity_error:.1f}m/s")
                
                # 计算奖励（误差越小奖励越高）
                range_accuracy = max(0, 1.0 - range_error / 20.0)  # 20米内完全准确
                velocity_accuracy = max(0, 1.0 - velocity_error / 2.0)  # 2m/s内完全准确
                
                accuracy_score = (range_accuracy + velocity_accuracy) / 2
                
                # RCS因子和距离衰减
                rcs = target.get('rcs', 1.0)
                range_factor = 1.0 / (1.0 + (true_range/1000)**2)  # 1000米参考距离
                
                # 调整奖励值为更合理的范围
                detection_bonus = 1.0  # 检测到目标的基础奖励（从5.0降低到1.0）
                accuracy_bonus = accuracy_score * 0.5  # 准确性奖励（从3.0降低到0.5）
                
                target_reward = detection_bonus + accuracy_bonus
                print(f"目标奖励: {target_reward:.3f} (基础: {detection_bonus:.1f}, 准确: {accuracy_bonus:.1f})")
                total_reward += target_reward
            else:
                print(f"⚠️ 未找到匹配的峰值 (最近距离: {best_distance:.1f}m)")
        
        # 显示所有检测到的峰值信息
        print(f"\n所有检测到的峰值:")
        for j, peak in enumerate(peaks):
            status = "已匹配" if j in matched_peaks else "未匹配"
            print(f"峰值 {j}: 距离={peak['range']:.1f}m, 速度={peak['velocity']:.1f}m/s, "
                f"强度={peak['intensity']:.1f}dB, {status}")
        
        # 额外奖励：检测到额外目标（误检惩罚较低）
        extra_detections = len(peaks) - len(matched_peaks)
        if extra_detections > 0:
            extra_reward = extra_detections * 0.1  # 较小的正奖励（从0.5降低到0.1）
            print(f"额外检测奖励: {extra_reward:.3f}")
            total_reward += extra_reward
        
        # 未检测目标的惩罚
        missed_targets = len(targets) - len(matched_targets)
        if missed_targets > 0:
            missed_penalty = missed_targets * -1.0  # 每个未检测目标惩罚（从-3.0降低到-1.0）
            print(f"未检测惩罚: {missed_penalty:.3f}")
            total_reward += missed_penalty
        
        print(f"总检测奖励: {total_reward:.3f}")
        print("="*50)
        return float(total_reward)

    def _estimate_noise_floor(self, rd_map):
        """
        鲁棒的距离-多普勒图噪声基准估计
        
        参数:
            rd_map: 距离-多普勒图（复数矩阵）
        
        返回:
            噪声水平(dB)
        """
        # 1. 将复数数据转换为幅度（或功率）
        magnitude = np.abs(rd_map)
        
        # 2. 排除强目标区域（使用百分位数阈值）
        # 假设噪声是幅度较小的部分
        threshold = np.percentile(magnitude, 90)  # 90%分位数作为阈值
        noise_mask = magnitude < threshold
        
        # 3. 计算噪声区域的统计量
        noise_samples = magnitude[noise_mask]
        
        if len(noise_samples) == 0:
            # 回退方案：使用整个矩阵的最小值
            return 10 * np.log10(np.min(magnitude) + 1e-9)
        
        # 4. 计算噪声基准（中值更鲁棒）
        noise_median = np.median(noise_samples)
        
        # 5. 转换为dB
        noise_db = 10 * np.log10(noise_median + 1e-9)
        
        return noise_db

    def _calculate_theoretical_signal(self, target, radar_params):
        """基于雷达方程计算理论信号强度"""
        # 雷达方程参数
        Pt = radar_params.get('tx_power', 100)  # 发射功率(W)
        G = radar_params.get('antenna_gain', 30)  # 天线增益(dB)
        λ = radar_params.get('wavelength', 0.0039)  # 波长(m) 77GHz雷达
        
        # 目标参数
        R = np.linalg.norm(target['location'])  # 距离(m)
        RCS = target.get('rcs', 1.0)  # 雷达截面积(m²)
        
        # 雷达方程（简化版）
        # Pr = (Pt * G² * λ² * RCS) / ((4π)³ * R⁴ * L)
        # 其中L是系统损耗（假设为1）
        
        # 计算接收功率
        Pr = (Pt * (10**(G/10))**2 * λ**2 * RCS) / \
            ((4 * np.pi)**3 * R**4)
        
        return Pr

    def _get_measured_signal(self, target, rd_map, radar_params):
        """获取目标位置的实际信号强度"""
        # 计算目标在RD图中的位置
        range_res = radar_params.get('range_resolution', 0.5)
        doppler_res = radar_params.get('doppler_resolution', 0.2)
        
        x, y, z = target['location']
        true_range = np.sqrt(x**2 + y**2 + z**2)
        range_bin = int(np.clip(true_range / range_res, 0, rd_map.shape[0]-1))
        
        # 计算径向速度
        radial_velocity = (
            target['speed'][0] * (x/true_range) +
            target['speed'][1] * (y/true_range) +
            target['speed'][2] * (z/true_range) if true_range > 0 else 0
        )
        
        doppler_bin = int(np.clip(
            radial_velocity / doppler_res, 
            0, 
            rd_map.shape[1]-1
        ))
        
        # 获取信号幅度
        return np.abs(rd_map[range_bin, doppler_bin])

    def _calculate_power_penalty(self, radar_params: Dict) -> float:
        """功率惩罚应为正值（越高表示功率浪费越多）"""
        # 功率指标（应为正值）
        gain = radar_params.get('gain', 0)
        tx_power = gain * 0.1  # 功率估算
        
        # 返回正值（表示需要惩罚的程度）
        return tx_power  # 这将乘以负权重变为负值

    def _calculate_interference_penalty(self, obs_dict: Dict) -> float:
        """计算干扰惩罚（分贝尺度+裁剪）"""
        processed_data = obs_dict['rd_map']
        
        # 获取幅度数据
        magnitude_data = np.abs(processed_data)
        
        # 转换为分贝尺度（避免log(0)）
        db_data = 10 * np.log10(magnitude_data + 1e-9)
        
        # 裁剪到合理范围（-100dB到0dB）
        clipped_db = np.clip(db_data, -100, 100)
        
        # 计算平均分贝值
        avg_db = np.mean(clipped_db)
        
        # 惩罚高干扰水平
        # 分贝值越高（越接近0），干扰越大，惩罚越大
        penalty = avg_db * 0.01  # 缩放因子
        
        # 确保惩罚值在合理范围
        return float(np.clip(penalty, 0, 10))

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
        reward -=  prf / max_prf

        # Penalty for long pulse width (reduced time resolution)
        pulse_width = radar_params.get(
            'pulse_width', self.pulse_width_range[0])
        max_pw = self.pulse_width_range[1]
        reward -= pulse_width / max_pw
        return float(reward)
    
    def _calculate_beam_reward(self, targets, radar_params):
        beam_az = radar_params['beam_az']
        beam_el = radar_params['beam_el']
        beam_width = 20.0  # 加宽波束至20°
        
        total_reward = 0
        for target in targets:
            x, y, z = target['location']
            
            # 计算目标角度（使用安全除法）
            r_xy = max(0.1, math.sqrt(x**2 + y**2))  # 避免除0
            target_az = math.degrees(math.atan2(y, x))
            target_el = math.degrees(math.atan2(z, r_xy))
            
            # 计算角度差异（考虑方位角周期性）
            az_diff = min(abs(target_az - beam_az), 360 - abs(target_az - beam_az))
            el_diff = abs(target_el - beam_el)
            
            # 使用高斯权重计算波束内贡献
            az_score = math.exp(-(az_diff**2)/(2*(beam_width/3)**2))  # 3σ=波束宽
            el_score = math.exp(-(el_diff**2)/(2*(beam_width/3)**2))
            in_beam_factor = az_score * el_score
            
            total_reward += in_beam_factor * target.get('rcs', 1.0)
        
        return total_reward    

    def _calculate_beam_reward2(self, targets: List[Dict], radar_params: Dict) -> float:
        """计算波束对准奖励（带详细调试）"""
        beam_az = radar_params.get('beam_az', 0)
        beam_el = radar_params.get('beam_el', 0)
        beam_width = 30.0  # 波束宽度（度）
        
        total_reward = 0
        
        print(f"波束方向: 方位角={beam_az:.1f}°, 俯仰角={beam_el:.1f}°, 波束宽度={beam_width}°")
        
        for i, target in enumerate(targets):
            # 计算目标位置
            x, y, z = target['location']
            
            # 计算目标距离
            distance = np.sqrt(x**2 + y**2 + z**2)
            
            # 计算目标方位角（从雷达视角）
            target_az = np.degrees(np.arctan2(y, x))
            
            # 计算目标俯仰角（从雷达视角）
            target_el = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
            
            # 计算角度差
            az_diff = abs(target_az - beam_az)
            el_diff = abs(target_el - beam_el)
            
            # 计算波束内比例
            in_beam_az = max(0, 1 - az_diff / (beam_width/2))
            in_beam_el = max(0, 1 - el_diff / (beam_width/2))
            in_beam_factor = in_beam_az * in_beam_el
            
            # 根据目标RCS加权
            rcs = target.get('rcs', 1.0)
            target_reward = in_beam_factor * rcs
            total_reward += target_reward
            
            # 详细调试输出
            print(f"目标 {i}: 位置=({x:.1f}, {y:.1f}, {z:.1f})m, "
                f"距离={distance:.1f}m, "
                f"方位角={target_az:.1f}°, 俯仰角={target_el:.1f}°, "
                f"方位差={az_diff:.1f}°, 俯仰差={el_diff:.1f}°, "
                f"波束内比例={in_beam_factor:.2f}, RCS={rcs:.1f}, 奖励={target_reward:.2f}")
        
        print(f"总波束对准奖励: {total_reward:.2f}")
        return total_reward

    def _get_targets_in_beam(self, targets: List[Dict], radar_params: Dict) -> list:
        """Get targets currently within any of the radar beams"""
        targets_in_beam = []
        
        # 获取波束参数并确保它们是 NumPy 数组
        beam_az = np.array(radar_params.get('beam_az', [0]))
        beam_el = np.array(radar_params.get('beam_el', [0]))
        
        beam_width = 10.0  # Degrees, default beam width
        
        # 如果 beam_az 和 beam_el 是标量，转换为数组
        if beam_az.ndim == 0:
            beam_az = np.array([beam_az])
        if beam_el.ndim == 0:
            beam_el = np.array([beam_el])
        
        # 确保两个数组长度相同
        if len(beam_az) != len(beam_el):
            # 如果长度不同，使用较短的长度
            min_len = min(len(beam_az), len(beam_el))
            beam_az = beam_az[:min_len]
            beam_el = beam_el[:min_len]
        
        for target in targets:
            # Extract target position
            x, y, z = target['location']
            
            # Calculate target angles relative to radar
            target_az = np.degrees(np.arctan2(y, x))
            target_el = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
            
            # 计算与所有波束的差异
            az_diffs = np.abs(target_az - beam_az)
            el_diffs = np.abs(target_el - beam_el)
            
            # 检查目标是否在任何波束内
            in_any_beam = np.any((az_diffs < beam_width/2) & (el_diffs < beam_width/2))
            
            if in_any_beam:
                targets_in_beam.append(target)

        return targets_in_beam


def main():
    import numpy as np
    from cognitive_radar.environment.radar_env import CognitiveRadarEnv

    # 配置环境
    config = {
        "radar_type": "PD-LS02",
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
        "targets": [
            {
                "model_type": MotionModelType.HIGH_SPEED_DRONE,
                "params": {
                    "start_position": [900, 50, 50],
                    "end_position": [1000, 200, 100],
                    "cruise_speed": 30,
                    "rcs": 0.5
                }
            },
            {
                "model_type": MotionModelType.HIGH_SPEED_DRONE,
                "params": {
                    "start_position": [1900, 50, 50],
                    "end_position": [2000, 200, 100],
                    "cruise_speed": -30,
                    "rcs": 0.5
                }
            }
        ]
    }

    # 创建环境
    env = CognitiveRadarEnv(config)

    # 重置环境
    obs, info = env.reset()
    print("初始观测类型:", type(obs))
    print("RD图形状:", obs['rd_map'].shape)
    print("特征维度:", obs['features'].shape)

    # 随机策略测试
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"奖励: {reward:.4f}, 终止: {terminated}, 截断: {truncated}")
        print(f"雷达参数: beam_az {info['beam_az']:.1f}°, prf {info['prf']:.1f}Hz")

        if terminated or truncated:
            break

    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()