from typing import Dict
import numpy as np
import radarsimpy as rp
from ..radar_system import RadarFactory, DefaultRadarFactory
from radarsimpy.simulator import sim_radar
import radarsimpy.processing as proc
from scipy.constants import speed_of_light
from scipy import signal
import matplotlib.pyplot as plt
from pprint import pprint


def normalize_rd_map(rd_map):
    """归一化距离-多普勒图"""
    max_val = np.max(np.abs(rd_map))
    if max_val > 0:
        return rd_map / max_val
    return rd_map


class RadarSimulator:
    """Wrapper for radar simulation using RadarSimPy"""

    def __init__(self, radar_type: str, params: Dict = {}):
        self.radar_type = radar_type
        self.radar = DefaultRadarFactory().create(radar_type)
        self.default_radar = DefaultRadarFactory().create(radar_type)
        self.last_simulation = None
        self.last_obs = None
        self.frame_results = {
            'baseband': [],
            'rd_maps': [],
            'observations': [],
            'peaks': [],
            'targets_history': []
        }        

    def reset_radar(self):
        self.radar = self.default_radar
        self.last_simulation = None
        self.last_obs = None
        
    def update_radar(self, params: Dict[str, float]):
        """更新雷达参数并验证有效性"""
        # 验证参数有效性
        if params['pulse_width'] <= 0:
            print(f"警告: 脉冲宽度无效 ({params['pulse_width']})")
            # params['pulse_width'] = self.pulse_width_range[0]
        
        if params['prf'] <= 0:
            print(f"警告: PRF无效 ({params['prf']})")
            # params['prf'] = self.prf_range[0]
        
        # # 确保频率在范围内
        # freq_min, freq_max = self.frequency_range
        # if params['frequency'] < freq_min or params['frequency'] > freq_max:
        #     print(f"警告: 频率无效 ({params['frequency']})，限制在范围内 {freq_min}-{freq_max}")
        #     params['frequency'] = np.clip(params['frequency'], freq_min, freq_max)
        
        # # 确保增益在范围内
        # gain_min, gain_max = self.gain_range
        # if params['gain'] < gain_min or params['gain'] > gain_max:
        #     print(f"警告: 增益无效 ({params['gain']})，限制在范围内 {gain_min}-{gain_max}")
        #     params['gain'] = np.clip(params['gain'], gain_min, gain_max)
        
        # 更新雷达
        self.radar = DefaultRadarFactory().create(self.radar_type, params=params)        

    def simulate(self, targets: list) -> np.ndarray:
        """Run radar simulation with current parameters"""
        data = sim_radar(
            self.radar,
            targets,
            density=2
        )

        self.last_simulation = data
        timestamp = data["timestamp"]
        baseband = data["baseband"]
        noise = data["noise"]
        return baseband + noise

    def process_signals(self, baseband: np.ndarray, window_type: str = "hamming") -> np.ndarray:
        """使用窗函数处理雷达信号"""

        # 计算每个脉冲的采样点数
        # type: ignore
        samples_per_pulse = self.radar.sample_prop["samples_per_pulse"] # type: ignore
        # type: ignore
        pulses = self.radar.radar_prop["transmitter"].waveform_prop["pulses"] # type: ignore

        # 根据选择的窗函数类型创建窗函数
        if window_type.lower() == "hamming":
            range_window = signal.windows.hamming(samples_per_pulse, sym=True)
            dop_window = signal.windows.hamming(pulses, sym=True)
        elif window_type.lower() == "hanning":
            range_window = signal.windows.hann(samples_per_pulse, sym=True)
            dop_window = signal.windows.hann(pulses, sym=True)
        elif window_type.lower() == "blackman":
            range_window = signal.windows.blackman(samples_per_pulse, sym=True)
            dop_window = signal.windows.blackman(pulses, sym=True)
        elif window_type.lower() == "chebyshev":
            range_window = signal.windows.chebwin(samples_per_pulse, at=60)
            dop_window = signal.windows.chebwin(pulses, at=60)
        elif window_type.lower() == "kaiser":
            range_window = signal.windows.kaiser(samples_per_pulse, beta=8)
            dop_window = signal.windows.kaiser(pulses, beta=8)
        elif window_type.lower() == "taylor":
            range_window = signal.windows.taylor(
                samples_per_pulse, nbar=3, sll=40)
            dop_window = signal.windows.taylor(pulses, nbar=3, sll=40)
        else:  # 矩形窗（无窗）
            range_window = None
            dop_window = None

        # 进行距离-多普勒FFT
        range_fft_points = samples_per_pulse
        rd_map = proc.range_doppler_fft(
            baseband,
            rwin=range_window,
            dwin=dop_window,
            rn=range_fft_points,
            dn=pulses
            )

        # 应用fftshift将零多普勒移到中心
        rd_map = np.fft.fftshift(rd_map, axes=1)

        return rd_map.squeeze(0)

    def compare_window_functions(self, baseband: np.ndarray):
        """比较不同窗函数的性能"""
        window_types = ["rectangular", "hamming", "hanning",
                        "blackman", "chebyshev", "kaiser", "taylor"]

        results = {}

        for window_type in window_types:
            # 处理信号
            rd_map = self.process_signals(baseband, window_type)

            self.plot_rd_map(
                rd_map=rd_map,
                title="Optimized Radar Range-Doppler Map",
                cmap="jet",
                save_path=f"optimized_rd_map_{window_type}.png",
                show=True
            )

            # 计算性能指标
            magnitude = np.abs(rd_map)
            db_magnitude = 10 * np.log10(magnitude + 1e-9)

            # 主瓣宽度（以-3dB点计算）
            max_val = np.max(db_magnitude)
            half_power = max_val - 3
            mask = db_magnitude >= half_power
            mainlobe_width = np.sum(mask) / np.sum(mask > 0)

            # 旁瓣水平（最高旁瓣与主瓣的差值）
            peak_idx = np.unravel_index(
                np.argmax(db_magnitude), db_magnitude.shape)
            sidelobe_mask = np.ones_like(db_magnitude, dtype=bool)
            sidelobe_mask[peak_idx[0]-2:peak_idx[0] +
                          3, peak_idx[1]-2:peak_idx[1]+3] = False
            max_sidelobe = np.max(db_magnitude[sidelobe_mask])
            sidelobe_level = max_val - max_sidelobe

            results[window_type] = {
                "mainlobe_width": mainlobe_width,
                "sidelobe_level": sidelobe_level,
                "peak_snr": max_val - np.median(db_magnitude)
            }

        return results

    def get_observation(self, baseband: np.ndarray) -> Dict:
        """Generate observation from raw radar data"""
        rd_map = self.process_signals(baseband)
        
        # 将复数转换为幅度（实数）
        rd_map = np.abs(rd_map)  # 或者使用 np.abs(rd_map)**2 表示功率        

        # 提取特征 - 获取最多3个峰值的特征
        features = self.extract_features(rd_map, num_peaks=3)

        self.last_obs = {
            'raw_data': baseband,
            'rd_map': rd_map.astype(np.float32),
            'features': features.astype(np.float32)
        }

        return self.last_obs

    def extract_features(self, rd_map: np.ndarray, num_peaks: int = 3) -> np.ndarray:
        """
        优化后的特征提取方法，只保留最有用的特征
        
        参数:
        - rd_map: 处理后的距离-多普勒图
        - num_peaks: 要提取的最大峰值数量
        
        返回:
        - 包含特征值的NumPy数组
        """
        # 计算幅度(dB尺度)
        magnitude = np.abs(rd_map)
        db_magnitude = 10 * np.log10(magnitude + 1e-9)  # 转换为dB尺度
        
        # 获取雷达参数
        params = self.get_current_radar_params()
        max_range = params['max_unambiguous_range']
        max_velocity = params['max_unambiguous_velocity']
        
        # 计算噪声基底
        noise_floor = np.percentile(db_magnitude, 25)
        
        # 检测峰值
        peaks = self.find_peaks(rd_map, num_peaks=num_peaks)
        
        # 提取峰值特征
        peak_features = []
        for i, peak in enumerate(peaks[:num_peaks]):
            peak_features.extend([
                peak['range'] / max_range,        # 归一化距离
                peak['velocity'] / max_velocity,  # 归一化速度
                peak['intensity'],                # 强度
                peak['snr'],                      # 信噪比
                peak['local_contrast']            # 局部对比度
            ])
        
        # 如果检测到的峰值少于请求的数量，用0填充
        while len(peak_features) < num_peaks * 5:
            peak_features.append(0.0)
        
        # 提取最有用的全局特征
        max_val = np.max(db_magnitude)
        mean_val = np.mean(db_magnitude)
        std_val = np.std(db_magnitude)
        
        # 计算信噪比分布
        snr_map = db_magnitude - noise_floor
        mean_snr = np.mean(snr_map[snr_map > 0])
        max_snr = np.max(snr_map)
        
        # 计算目标密度（使用峰值数量）
        target_density = len(peaks) / (rd_map.shape[0] * rd_map.shape[1])
        
        # 计算信号动态范围
        dynamic_range = max_val - noise_floor
        
        # 计算信号峰均比（PAPR）
        papr = max_val - mean_val
        
        # 组合最有用的特征
        features = [
            mean_val,           # 平均值 (dB)
            std_val,            # 标准差 (dB)
            noise_floor,        # 噪声基底 (dB)
            mean_snr,           # 平均信噪比 (dB)
            max_snr,            # 最大信噪比 (dB)
            dynamic_range,      # 动态范围 (dB)
            papr,               # 峰均比 (dB)
            target_density      # 目标密度（基于峰值数量）
        ]
        
        # 添加峰值特征
        features.extend(peak_features)
        
        return np.array(features)

    def get_current_radar_params(self) -> Dict:
        params = {}

        # 获取基本参数
        radar_prop = self.radar.radar_prop  # type: ignore
        params['beam_az'] = radar_prop['transmitter'].txchannel_prop["az_angles"]
        params['beam_el'] = radar_prop['transmitter'].txchannel_prop["el_angles"]

        params['gain'] = radar_prop['receiver'].rxchannel_prop["antenna_gains"]

        params['bandwidth'] = radar_prop['transmitter'].waveform_prop["bandwidth"]
        params['frequency'] = radar_prop['transmitter'].waveform_prop["f"]

        params['pulse_width'] = radar_prop['transmitter'].waveform_prop["pulse_length"]
        prp = radar_prop['transmitter'].waveform_prop["prp"][0]
        prf = 1 / prp
        params['prf'] = prf
        params['tx_power'] = radar_prop['transmitter'].rf_prop["tx_power"]

        # 计算波长 - 使用中心频率而不是平均频率
        center_freq = np.mean(params['frequency'])
        wavelength = speed_of_light / center_freq
        pulses = radar_prop['transmitter'].waveform_prop["pulses"]

        # 正确的速度分辨率计算
        params['pulses'] = pulses
        params['velocity_resolution'] = wavelength * prf / (2 * pulses)

        # 其他参数
        params['range_resolution'] = speed_of_light / (2 * params['bandwidth'])
        params['max_unambiguous_range'] = speed_of_light / (2 * prf)
        params['max_unambiguous_velocity'] = wavelength * prf / 4

        # 添加采样率和采样点数
        params['sampling_rate'] = radar_prop['receiver'].bb_prop["fs"]
        # type: ignore
        params['samples_per_pulse'] = self.radar.sample_prop["samples_per_pulse"] # type: ignore

        # 添加中心频率和波长
        params['center_frequency'] = center_freq
        params['wavelength'] = wavelength
        return params
    
    def step(self, targets: list, time_step: float = 0.1, collect_results: bool = True) -> Dict[str, any]: # type: ignore
        """
        执行单帧雷达仿真，更新目标位置和速度
        
        参数:
        - targets: 当前帧的目标列表，每个目标包含位置、速度等信息
        - time_step: 时间步长（秒）
        - collect_results: 是否收集结果用于后续分析
        
        返回:
        - Dict[str, any]: 包含当前帧的仿真结果
        """
        # 更新目标位置（基于速度和加速度）
        updated_targets = self._update_targets_position(targets, time_step)
        
        # 模拟雷达信号
        baseband = self.simulate(updated_targets)
        
        # 处理信号
        rd_map = self.process_signals(baseband)
        
        # 获取观测结果
        observation = self.get_observation(baseband)
        
        # 检测峰值
        peaks = self.find_peaks(rd_map)
        
        # 收集结果（如果启用）
        if collect_results:
            self.frame_results['baseband'].append(baseband)
            self.frame_results['rd_maps'].append(rd_map)
            self.frame_results['observations'].append(observation)
            self.frame_results['peaks'].append(peaks)
            self.frame_results['targets_history'].append(updated_targets.copy())        
        
        # 返回当前帧结果和更新后的目标
        frame_result = {
            'baseband': baseband,
            'rd_map': rd_map,
            'observation': observation,
            'peaks': peaks,
            'updated_targets': updated_targets
        }
        
        return frame_result
    
    def clear_results(self):
        """清空收集的结果"""
        self.frame_results = {
            'baseband': [],
            'rd_maps': [],
            'observations': [],
            'peaks': [],
            'targets_history': []
        }  
    def get_collected_results(self) -> Dict[str, list]:
        """获取收集的所有帧结果"""
        return self.frame_results   
    
    def plot_trajectory_from_collected(self, save_path: str = None, show: bool = True): # type: ignore
        """
        使用收集的结果绘制航迹对比图
        """
        if not self.frame_results['targets_history']:
            print("No collected results available. Run step() with collect_results=True first.")
            return None
        
        return self.plot_trajectory_comparison(self.frame_results, save_path, show)           

    def _update_targets_position(self, targets: list, time_step: float) -> list:
        """
        根据目标的速度和加速度更新目标位置
        
        参数:
        - targets: 当前帧的目标列表
        - time_step: 时间步长（秒）
        
        返回:
        - list: 更新后的目标列表
        """
        updated_targets = []
        
        for target in targets:
            # 复制目标以避免修改原始数据
            new_target = target.copy()
            
            # 提取当前位置和速度
            location = np.array(target['location'])
            speed = np.array(target.get('speed', (0, 0, 0)))
            acceleration = np.array(target.get('acceleration', (0, 0, 0)))
            
            # 更新速度（考虑加速度）
            new_speed = speed + acceleration * time_step
            
            # 更新位置
            new_location = location + new_speed * time_step
            
            # 更新目标信息
            new_target['location'] = tuple(new_location)
            new_target['speed'] = tuple(new_speed)
            
            updated_targets.append(new_target)
        
        return updated_targets

    def simulate_multiple_frames(self, initial_targets: list, frame_count: int = 10, 
                            time_step: float = 0.1) -> Dict[str, list]:
        """
        执行多帧仿真（使用step函数）
        
        参数:
        - initial_targets: 初始目标列表
        - frame_count: 仿真帧数
        - time_step: 每帧时间步长（秒）
        
        返回:
        - Dict[str, list]: 包含所有帧的结果
        """
        all_results = {
            'baseband': [],
            'rd_maps': [],
            'observations': [],
            'peaks': [],
            'targets_history': []
        }
        
        current_targets = initial_targets.copy()
        
        for frame in range(frame_count):
            print(f"Processing frame {frame + 1}/{frame_count}")
            
            # 执行单帧步进
            frame_result = self.step(current_targets, time_step)
            
            # 保存结果
            all_results['baseband'].append(frame_result['baseband'])
            all_results['rd_maps'].append(frame_result['rd_map'])
            all_results['observations'].append(frame_result['observation'])
            all_results['peaks'].append(frame_result['peaks'])
            all_results['targets_history'].append(frame_result['updated_targets'])
            
            # 更新目标用于下一帧
            current_targets = frame_result['updated_targets']
        
        return all_results

    def add_acceleration_to_targets(self, targets: list, acceleration: tuple = (0, 0, 0)):
        """
        为目标添加加速度信息
        
        参数:
        - targets: 目标列表
        - acceleration: 加速度向量 (ax, ay, az)
        
        返回:
        - list: 添加加速度后的目标列表
        """
        accelerated_targets = []
        for target in targets:
            new_target = target.copy()
            new_target['acceleration'] = acceleration
            accelerated_targets.append(new_target)
        
        return accelerated_targets    

    def find_peaks(self, rd_map, size=9, threshold=10, num_peaks=10):
        """
        检测二维数据的峰值
        :param rd_map: 二维雷达数据（距离-多普勒图）
        :param size: 邻域窗口大小（奇数）
        :param threshold: 峰值强度阈值
        :param num_peaks: 可检测的峰值数量
        :return: 峰值信息列表，包含位置、距离、速度、强度等信息
        """
        from skimage.feature import peak_local_max

        # 计算幅度(dB尺度)
        magnitude = np.abs(rd_map)
        db_magnitude = 10 * np.log10(magnitude + 1e-9)  # 转换为dB尺度

        # 计算噪声基底和自适应阈值
        noise_floor = np.percentile(db_magnitude, 25)

        threshold_abs = noise_floor + \
            threshold if threshold else np.percentile(db_magnitude, 25)

        # 峰值检测
        # 使用自定义邻域形状
        footprint = np.ones((3, 3), dtype=bool)  # 3x3正方形邻域
        peaks = peak_local_max(db_magnitude,
                               min_distance=size,
                               threshold_abs=threshold_abs, # 绝对强度阈值
                               threshold_rel=0.45,  # 相对于最大强度设置阈值
                               # footprint=footprint, # 自定义邻域形状
                               num_peaks=num_peaks)  # 增加可检测的峰值数量

        # 转换为物理坐标
        params = self.get_current_radar_params()
        range_resolution = params['range_resolution']
        velocity_resolution = params['velocity_resolution']

        num_pulses, num_range_bins = rd_map.shape
        ranges = np.arange(0, num_range_bins) * range_resolution
        velocities = np.linspace(-num_pulses / 2,
                                 num_pulses / 2,
                                 num_pulses) * velocity_resolution

        # 峰值验证和后处理
        peak_info = []
        for peak in peaks:
            row, col = peak[0], peak[1]
            intensity = round(float(db_magnitude[row, col]),1)

            # 计算局部对比度
            # 定义区域半径 ,通常，5×5 区域是一个良好的起点，它在计算效率和统计稳定性之间提供了良好的平衡。
            row_radius = 2  # 行方向半径
            col_radius = 2  # 列方向半径

            local_region = db_magnitude[
                max(0, row - row_radius):min(row+row_radius + 1, db_magnitude.shape[0]),
                max(0, col - col_radius):min(col+col_radius + 1, db_magnitude.shape[1])
            ]
            local_mean = np.mean(local_region)
            local_contrast = round(float(intensity - local_mean),1)

            # 只保留对比度足够的峰值
            if local_contrast > threshold * 0.5:
                range_val = round(float(ranges[col]),1)
                velocity_val = round(float(velocities[row]),1)
                peak_info.append({
                    'position': (int(row), int(col)),
                    'range': range_val,  # 保持浮点精度
                    'velocity': velocity_val,  # 保持浮点精度
                    'intensity': intensity,
                    'snr': round(float(intensity - noise_floor),1),
                    'local_contrast': local_contrast
                })

        # 按强度排序
        peak_info.sort(key=lambda x: x['intensity'], reverse=True)

        return peak_info
    
    def plot_trajectory_comparison(self, results: Dict[str, list], save_path: str = None, show: bool = True): # type: ignore
        """
        绘制目标实际航迹和检测航迹对比图
        
        参数:
        - results: simulate_multiple_frames返回的结果字典
        - save_path: 保存路径
        - show: 是否显示图表
        """
        frame_count = len(results['targets_history'])
        time_steps = np.arange(frame_count)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 提取实际目标信息
        actual_ranges = []  # 实际距离
        actual_velocities = []  # 实际速度
        target_count = len(results['targets_history'][0])
        
        for frame_targets in results['targets_history']:
            frame_ranges = []
            frame_velocities = []
            for target in frame_targets:
                # 计算实际距离（假设雷达在原点）
                distance = np.sqrt(target['location'][0]**2 + 
                                target['location'][1]**2 + 
                                target['location'][2]**2)
                frame_ranges.append(distance)
                
                # 计算径向速度（假设雷达在原点看向x方向）
                radial_speed = target['speed'][0]  # 简化假设
                frame_velocities.append(radial_speed)
            
            actual_ranges.append(frame_ranges)
            actual_velocities.append(frame_velocities)
        
        # 提取检测到的目标信息
        detected_ranges = []  # 检测距离
        detected_velocities = []  # 检测速度
        
        for frame_peaks in results['peaks']:
            frame_detected_ranges = []
            frame_detected_velocities = []
            
            # 对每个实际目标，找到最接近的检测目标
            for target_idx in range(target_count):
                closest_range = None
                closest_velocity = None
                min_distance = float('inf')
                
                for peak in frame_peaks:
                    # 计算检测目标与实际目标的匹配度
                    range_diff = abs(peak['range'] - actual_ranges[-1][target_idx])
                    velocity_diff = abs(peak['velocity'] - actual_velocities[-1][target_idx])
                    total_diff = range_diff + velocity_diff * 0.1  # 加权
                    
                    if total_diff < min_distance:
                        min_distance = total_diff
                        closest_range = peak['range']
                        closest_velocity = peak['velocity']
                
                frame_detected_ranges.append(closest_range)
                frame_detected_velocities.append(closest_velocity)
            
            detected_ranges.append(frame_detected_ranges)
            detected_velocities.append(frame_detected_velocities)
        
        # 绘制距离对比图
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        markers = ['o', 's', '^', 'D', 'v']
        
        for target_idx in range(target_count):
            # 实际航迹
            actual_range_data = [r[target_idx] for r in actual_ranges]
            ax1.plot(time_steps, actual_range_data, 
                    color=colors[target_idx % len(colors)], 
                    marker=markers[target_idx % len(markers)],
                    linestyle='-', linewidth=2, markersize=6,
                    label=f'Target {target_idx+1} Actual')
            
            # 检测航迹
            detected_range_data = [r[target_idx] for r in detected_ranges if r[target_idx] is not None]
            valid_time_steps = [t for t, r in zip(time_steps, detected_ranges) if r[target_idx] is not None]
            
            if detected_range_data:
                ax1.plot(valid_time_steps, detected_range_data,
                        color=colors[target_idx % len(colors)],
                        marker=markers[target_idx % len(markers)],
                        linestyle='--', linewidth=2, markersize=8,
                        markerfacecolor='white', markeredgewidth=2,
                        label=f'Target {target_idx+1} Detected')
        
        ax1.set_xlabel('Frame Number', fontsize=12)
        ax1.set_ylabel('Range (m)', fontsize=12)
        ax1.set_title('Range Trajectory: Actual vs Detected', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制速度对比图
        for target_idx in range(target_count):
            # 实际速度
            actual_velocity_data = [v[target_idx] for v in actual_velocities]
            ax2.plot(time_steps, actual_velocity_data,
                    color=colors[target_idx % len(colors)],
                    marker=markers[target_idx % len(markers)],
                    linestyle='-', linewidth=2, markersize=6,
                    label=f'Target {target_idx+1} Actual')
            
            # 检测速度
            detected_velocity_data = [v[target_idx] for v in detected_velocities if v[target_idx] is not None]
            valid_time_steps = [t for t, v in zip(time_steps, detected_velocities) if v[target_idx] is not None]
            
            if detected_velocity_data:
                ax2.plot(valid_time_steps, detected_velocity_data,
                        color=colors[target_idx % len(colors)],
                        marker=markers[target_idx % len(markers)],
                        linestyle='--', linewidth=2, markersize=8,
                        markerfacecolor='white', markeredgewidth=2,
                        label=f'Target {target_idx+1} Detected')
        
        ax2.set_xlabel('Frame Number', fontsize=12)
        ax2.set_ylabel('Velocity (m/s)', fontsize=12)
        ax2.set_title('Velocity Trajectory: Actual vs Detected', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 计算并显示性能指标
        self._display_performance_metrics(actual_ranges, detected_ranges, 
                                        actual_velocities, detected_velocities)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory comparison plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig

    def _display_performance_metrics(self, actual_ranges, detected_ranges, 
                                actual_velocities, detected_velocities):
        """
        显示检测性能指标
        """
        print("\n" + "="*50)
        print("Detection Performance Metrics")
        print("="*50)
        
        target_count = len(actual_ranges[0])
        frame_count = len(actual_ranges)
        
        for target_idx in range(target_count):
            range_errors = []
            velocity_errors = []
            detection_count = 0
            
            for frame in range(frame_count):
                if (detected_ranges[frame][target_idx] is not None and 
                    detected_velocities[frame][target_idx] is not None):
                    
                    range_error = abs(detected_ranges[frame][target_idx] - actual_ranges[frame][target_idx])
                    velocity_error = abs(detected_velocities[frame][target_idx] - actual_velocities[frame][target_idx])
                    
                    range_errors.append(range_error)
                    velocity_errors.append(velocity_error)
                    detection_count += 1
            
            if range_errors:
                avg_range_error = np.mean(range_errors)
                avg_velocity_error = np.mean(velocity_errors)
                detection_rate = detection_count / frame_count * 100
                
                print(f"Target {target_idx+1}:")
                print(f"  Detection Rate: {detection_rate:.1f}%")
                print(f"  Avg Range Error: {avg_range_error:.2f} m")
                print(f"  Avg Velocity Error: {avg_velocity_error:.2f} m/s")
                print(f"  Max Range Error: {max(range_errors):.2f} m")
                print(f"  Max Velocity Error: {max(velocity_errors):.2f} m/s")
            else:
                print(f"Target {target_idx+1}: No detections")
            
            print("-" * 30)

    def plot_2d_trajectory(self, results: Dict[str, list], save_path: str = None, show: bool = True): # type: ignore
        """
        绘制2D空间轨迹图（X-Y平面）
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 提取目标位置历史
        target_count = len(results['targets_history'][0])
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        print( target_count)
        for target_idx in range(target_count):
            x_positions = []
            y_positions = []
            
            for frame_targets in results['targets_history']:
                target = frame_targets[target_idx]
                x_positions.append(target['location'][0])
                y_positions.append(target['location'][1])
            
            # 绘制实际轨迹
            ax.plot(x_positions, y_positions, 
                color=colors[target_idx % len(colors)],
                marker='o', markersize=6, linestyle='-',
                linewidth=2, label=f'Target {target_idx+1} Actual')
            
            # 标记起点和终点
            ax.plot(x_positions[0], y_positions[0], 'o', 
                markersize=10, color=colors[target_idx % len(colors)],
                markerfacecolor='white', markeredgewidth=2)
            ax.plot(x_positions[-1], y_positions[-1], 's', 
                markersize=10, color=colors[target_idx % len(colors)],
                markerfacecolor='white', markeredgewidth=2)
        
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title('2D Spatial Trajectory', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig    

    def smart_colorbar_ticks(self, data, vmin=None, vmax=None):
        """
        智能选择颜色条刻度
        """
        if vmin is None:
            vmin = np.min(data)
        if vmax is None:
            vmax = np.max(data)

        # 计算数据范围
        data_range = vmax - vmin

        # 根据范围选择刻度间隔
        if data_range > 60:
            step = 10
        elif data_range > 30:
            step = 5
        elif data_range > 15:
            step = 2
        else:
            step = 1

        # 确保刻度从整数开始
        start = np.floor(vmin / step) * step
        end = np.ceil(vmax / step) * step

        # 生成刻度
        ticks = np.arange(start, end + step, step)

        # 过滤超出范围的刻度
        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]

        # 生成标签
        tick_labels = []
        for tick in ticks:
            if tick.is_integer():
                tick_labels.append(f"{int(tick)}")
            else:
                tick_labels.append(f"{tick:.1f}")

        return ticks, tick_labels

    def plot_rd_map(self, rd_map: np.ndarray = None, title: str = "Optimized Radar Range-Doppler Map",  # type: ignore
                    figsize: tuple = (12, 8), cmap: str = "jet",
                    save_path: str = None, show: bool = True):  # type: ignore
        """
        绘制专业级距离-多普勒图(RD图)
        """
        import matplotlib.pyplot as plt

        if rd_map is None and self.last_obs is not None:
            rd_map = self.last_obs['rd_map']
        elif rd_map is None:
            raise ValueError("No RD map available to plot")

        # 获取雷达参数
        params = self.get_current_radar_params()

        # 使用计算出的最大不模糊速度
        max_doppler_velocity = params['max_unambiguous_velocity']
        max_range = 3000

        # 计算波长
        wavelength = params['wavelength']

        # 计算距离和速度分辨率
        range_resolution = params['range_resolution']
        velocity_resolution = params['velocity_resolution']

        # 计算幅度(取绝对值)
        magnitude = np.abs(rd_map)

        # 创建图表和坐标轴
        fig, ax = plt.subplots(figsize=figsize)

        # 使用对数尺度显示(更好地展示动态范围)
        log_magnitude = 10 * np.log10(magnitude + 1e-9)  # 转换为dB尺度

        vmin = np.min(log_magnitude)
        vmax = np.max(log_magnitude)

        # 创建热图 - 使用物理单位作为范围
        # 注意：由于我们使用了fftshift，速度轴应该从负最大速度到正最大速度
        im = ax.imshow(log_magnitude,
                       aspect='auto',
                       cmap=cmap,
                       origin='lower',
                       interpolation='nearest',
                       vmin=vmin,
                       vmax=vmax,
                       extent=[0, max_range, -max_doppler_velocity, max_doppler_velocity])  # type: ignore

        # 添加颜色条 计算数据的动态范围
        ticks, tick_labels = self.smart_colorbar_ticks(
            log_magnitude, vmin=vmin, vmax=vmax)

        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Magnitude (dB)', fontsize=12)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)

        # 设置坐标轴标签
        ax.set_xlabel('Range (m)', fontsize=12)
        ax.set_ylabel('Velocity (m/s)', fontsize=12)

        x_ticks = np.arange(0, max_range + 1, 500)
        x_labels = [str(int(tick)) for tick in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

        y_ticks = np.arange(-max_doppler_velocity,
                            max_doppler_velocity + 1, 10)
        y_labels = [f"{y:.1f}" for y in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')

        # 添加标题
        ax.set_title(title, fontsize=14, pad=20)

        # 添加零多普勒线
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.8)
        ax.text(max_range * 0.02, max_doppler_velocity * 0.02,
                'Zero Doppler', color='r', fontsize=11, weight='bold')

        # 添加距离刻度标记 - 每100米标记一次
        for distance in range(0, int(max_range) + 100, 100):
            if distance <= max_range:
                ax.axvline(x=distance, color='white', linestyle=':', alpha=0.5)

        # 添加雷达参数信息 - 与图片完全一致
        param_text = (
            f"fc: {params['center_frequency']/1e9:.1f} GHz\n"
            f"Bandwidth: {params['bandwidth']/1e6:.1f} MHz\n"
            f"pulse_width: {params['pulse_width'] * 1e6:.1f} us\n"
            f"PRF: {params['prf']/1e3:.1f} kHz\n"
            f"Pulses: {params['pulses']} \n"
            f"tx_power: {params['tx_power']} W\n"
            f"sampling_rate: {params['sampling_rate']/1e6:.1f} MHz\n"
            f"samples_per_pulse: {params['samples_per_pulse']} \n"
            f"wavelength: {params['wavelength']:.3f} m\n"
            f"max_unambiguous_range: {params['max_unambiguous_range']/1000:.1f} KM\n"
            f"max_unambiguous_velocity: {params['max_unambiguous_velocity']:.1f} m/s\n"
            f"Range Res: {range_resolution:.2f} m\n"
            f"Velocity Res: {velocity_resolution:.2f} m/s"
        )
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

        # 优化布局
        plt.tight_layout()

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # 显示图像
        if show:
            plt.show()

        return fig
    
    def generate_rd_animation(
        self,
        results: Dict[str, list],
        output_path: str = "rd_animation.gif",
        fps: int = 2,
        dpi: int = 100,
        cmap: str = "jet",
        show_progress: bool = True
    ):
        """
        生成RD图动画(GIF/MP4)
        
        参数:
        - results: simulate_multiple_frames返回的结果字典
        - output_path: 输出文件路径(.gif或.mp4)
        - fps: 帧率
        - dpi: 图像分辨率
        - cmap: 颜色映射
        - show_progress: 是否显示进度
        """
        import matplotlib.animation as animation
        from matplotlib.animation import FFMpegWriter, PillowWriter
        import os        
        if not results['rd_maps']:
            raise ValueError("No RD maps found in results")

        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.close()  # 防止重复显示静态图

        # 初始化图像
        first_rd = results['rd_maps'][0]
        magnitude = np.abs(first_rd)
        db_magnitude = 10 * np.log10(magnitude + 1e-9)
        im = ax.imshow(db_magnitude, aspect='auto', cmap=cmap, origin='lower')
        
        # 设置坐标轴
        params = self.get_current_radar_params()
        max_velocity = params['max_unambiguous_velocity']
        max_range = 3000  # 可根据需要调整
        
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Range-Doppler Map Animation')
        
        # 设置颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Magnitude (dB)')

        # 动画更新函数
        def update(frame):
            rd_map = results['rd_maps'][frame]
            magnitude = np.abs(rd_map)
            db_magnitude = 10 * np.log10(magnitude + 1e-9)
            
            im.set_array(db_magnitude)
            im.set_clim(vmin=np.min(db_magnitude), vmax=np.max(db_magnitude))
            
            ax.set_title(f'Frame {frame + 1}/{len(results["rd_maps"])}')
            
            if show_progress:
                print(f"Processing frame {frame + 1}/{len(results['rd_maps'])}")
            
            return [im]

        # 创建动画
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(results['rd_maps']),
            interval=1000/fps,
            blit=True
        )

        # 保存动画
        output_ext = os.path.splitext(output_path)[1].lower()
        
        if output_ext == '.gif':
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer, dpi=dpi)
        elif output_ext == '.mp4':
            writer = FFMpegWriter(fps=fps, metadata=dict(title='RD Map Animation'))
            anim.save(output_path, writer=writer, dpi=dpi)
        else:
            raise ValueError("Unsupported output format. Use .gif or .mp4")

        print(f"Animation saved to {output_path}")
        return anim    

    def plot_3d_rd_map(self, rd_map: np.ndarray = None, title: str = "3D Range-Doppler Map",  # type: ignore
                       figsize: tuple = (14, 10), cmap: str = "jet",
                       elevation: float = 30, azimuth: float = 45,
                       save_path: str = None, show: bool = True):  # type: ignore
        """
        绘制三维距离-多普勒图，提供更直观的信号强度可视化

        参数:
        - rd_map: 距离多普勒图数据，如果为None则使用last_obs中的数据
        - title: 图表标题
        - figsize: 图表尺寸
        - cmap: 颜色映射
        - elevation: 3D视图的仰角
        - azimuth: 3D视图的方位角
        - save_path: 保存路径，如果为None则不保存
        - show: 是否显示图表
        """
        if rd_map is None and self.last_obs is not None:
            rd_map = self.last_obs['rd_map']
        elif rd_map is None:
            raise ValueError("No RD map available to plot")

        # 获取雷达参数
        params = self.get_current_radar_params()

        # 计算距离和多普勒轴
        range_resolution = params['range_resolution']
        velocity_resolution = params['velocity_resolution']
        max_unambiguous_range = 3000  # params['max_unambiguous_range']
        max_unambiguous_velocity = params['max_unambiguous_velocity']

        # 创建距离轴和多普勒轴
        range_bins = rd_map.shape[0]
        doppler_bins = rd_map.shape[1]

        range_axis = np.linspace(0, max_unambiguous_range, range_bins)
        doppler_axis = np.linspace(-max_unambiguous_velocity,
                                   max_unambiguous_velocity, doppler_bins)

        # 创建网格
        R, V = np.meshgrid(range_axis, doppler_axis, indexing='ij')

        # 计算幅度(dB尺度)
        magnitude = np.abs(rd_map)
        db_magnitude = 10 * np.log10(magnitude + 1e-9)  # 避免log(0)

        # 创建3D图形
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # 绘制表面图
        surf = ax.plot_surface(R, V, db_magnitude, cmap=cmap,
                               linewidth=0, antialiased=True, alpha=0.8)

        # 添加颜色条
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Magnitude (dB)', fontsize=12)

        # 设置坐标轴标签
        ax.set_xlabel('Range (m)', fontsize=12, labelpad=10)
        ax.set_ylabel('Velocity (m/s)', fontsize=12, labelpad=10)
        ax.set_zlabel('Magnitude (dB)', fontsize=12, labelpad=10)

        # 设置视角
        ax.view_init(elev=elevation, azim=azimuth)

        # 添加标题
        ax.set_title(title, fontsize=14, pad=20)

        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.3)

        # 添加雷达参数信息
        param_text = (
            f"Bandwidth: {params['bandwidth']/1e6:.1f} MHz\n"
            f"PRF: {params['prf']/1e3:.1f} kHz\n"
            f"Pulses: {params['pulses']}\n"
            f"Range Res: {range_resolution:.2f} m\n"
            f"Velocity Res: {velocity_resolution:.2f} m/s"
        )
        ax.text2D(0.02, 0.98, param_text, transform=ax.transAxes,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 优化布局
        plt.tight_layout()

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D RD map saved to {save_path}")

        # 显示图像
        if show:
            plt.show()

        return fig

    def plot_3d_rd_map_contour(self, rd_map: np.ndarray = None, title: str = "3D Range-Doppler Contour Map",  # type: ignore
                               figsize: tuple = (14, 10), cmap: str = "jet",
                               elevation: float = 30, azimuth: float = 45,
                               save_path: str = None, show: bool = True):  # type: ignore
        """
        绘制三维距离-多普勒等高线图，提供另一种视角

        参数与plot_3d_rd_map相同
        """
        if rd_map is None and self.last_obs is not None:
            rd_map = self.last_obs['rd_map']
        elif rd_map is None:
            raise ValueError("No RD map available to plot")

        # 获取雷达参数
        params = self.get_current_radar_params()

        # 计算距离和多普勒轴
        range_resolution = params['range_resolution']
        velocity_resolution = params['velocity_resolution']
        max_unambiguous_range = 3000  # params['max_unambiguous_range']
        max_unambiguous_velocity = params['max_unambiguous_velocity']

        # 创建距离轴和多普勒轴
        range_bins = rd_map.shape[0]
        doppler_bins = rd_map.shape[1]

        range_axis = np.linspace(0, max_unambiguous_range, range_bins)
        doppler_axis = np.linspace(-max_unambiguous_velocity,
                                   max_unambiguous_velocity, doppler_bins)

        # 创建网格
        R, V = np.meshgrid(range_axis, doppler_axis, indexing='ij')

        # 计算幅度(dB尺度)
        magnitude = np.abs(rd_map)
        db_magnitude = 10 * np.log10(magnitude + 1e-9)  # 避免log(0)

        # 创建3D图形
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # 绘制等高线图
        contour = ax.contour3D(R, V, db_magnitude, 50, cmap=cmap, alpha=0.8)

        # 添加颜色条
        cbar = fig.colorbar(contour, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Magnitude (dB)', fontsize=12)

        # 设置坐标轴标签
        ax.set_xlabel('Range (m)', fontsize=12, labelpad=10)
        ax.set_ylabel('Velocity (m/s)', fontsize=12, labelpad=10)
        ax.set_zlabel('Magnitude (dB)', fontsize=12, labelpad=10)

        # 设置视角
        ax.view_init(elev=elevation, azim=azimuth)

        # 添加标题
        ax.set_title(title, fontsize=14, pad=20)

        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.3)

        # 添加雷达参数信息
        param_text = (
            f"Bandwidth: {params['bandwidth']/1e6:.1f} MHz\n"
            f"PRF: {params['prf']/1e3:.1f} kHz\n"
            f"Pulses: {params['pulses']}\n"
            f"Range Res: {range_resolution:.2f} m\n"
            f"Velocity Res: {velocity_resolution:.2f} m/s"
        )
        ax.text2D(0.02, 0.98, param_text, transform=ax.transAxes,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 优化布局
        plt.tight_layout()

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D RD contour map saved to {save_path}")

        # 显示图像
        if show:
            plt.show()

        return fig
    
    def plot_range_pulse_map(self, baseband: np.ndarray = None, title: str = "Range-Pulse Map",  # type: ignore
                            figsize: tuple = (12, 8), cmap: str = "jet",
                            save_path: str = None, show: bool = True):  # type: ignore
        """
        绘制距离-脉冲序号图（RP图）
        横轴：距离（米）
        纵轴：脉冲序号
        
        参数:
        - baseband: 基带信号数据，如果为None则使用last_simulation中的数据
        - title: 图表标题
        - figsize: 图表尺寸
        - cmap: 颜色映射
        - save_path: 保存路径
        - show: 是否显示图表
        """
        if baseband is None and self.last_simulation is not None:
            baseband = self.last_simulation["baseband"]
        elif baseband is None:
            raise ValueError("No baseband data available to plot")
        
        # 获取雷达参数
        params = self.get_current_radar_params()
        
        # 计算每个脉冲的采样点数
        samples_per_pulse = params['samples_per_pulse']
        pulses = params['pulses']
        
        # 确保数据形状正确
        if baseband.ndim == 3:
            baseband = baseband.squeeze(0)  # 移除通道维度
        
        # 对每个脉冲进行距离FFT
        range_profiles = []
        for pulse_idx in range(pulses):
            # 提取当前脉冲的数据
            pulse_data = baseband[pulse_idx, :]
            
            # 进行距离FFT
            range_fft = np.fft.fft(pulse_data, n=samples_per_pulse)
            range_profile = np.abs(range_fft)
            
            range_profiles.append(range_profile)
        
        # 转换为数组（脉冲×距离）
        range_pulse_map = np.array(range_profiles)  # 脉冲在行，距离在列
        
        # 计算距离分辨率
        range_resolution = params['range_resolution']
        max_range = samples_per_pulse * range_resolution
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 使用对数尺度显示
        log_map = 10 * np.log10(range_pulse_map + 1e-9)  # 转换为dB尺度
        
        # 创建热图 - 横轴是距离，纵轴是脉冲序号
        im = ax.imshow(log_map,
                    aspect='auto',
                    cmap=cmap,
                    origin='lower',
                    interpolation='nearest',
                    extent=[0, max_range, 0, pulses])  # type: ignore
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Magnitude (dB)', fontsize=12)
        
        # 设置坐标轴标签
        ax.set_xlabel('Range (m)', fontsize=12)
        ax.set_ylabel('Pulse Number', fontsize=12)
        
        # 设置x轴刻度（距离）
        x_ticks = np.arange(0, max_range + 1, 500)
        x_labels = [str(int(tick)) for tick in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        
        # 设置y轴刻度（脉冲序号）
        y_ticks = np.arange(0, pulses + 1, max(1, pulses // 10))
        ax.set_yticks(y_ticks)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # 添加标题
        ax.set_title(title, fontsize=14, pad=20)
        
        # 添加雷达参数信息
        param_text = (
            f"Range Resolution: {range_resolution:.2f} m\n"
            f"Pulses: {pulses}\n"
            f"Samples per Pulse: {samples_per_pulse}\n"
            f"Max Range: {max_range:.1f} m"
        )
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
        
        # 优化布局
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Range-Pulse map saved to {save_path}")
        
        # 显示图像
        if show:
            plt.show()
        
        return fig

def interactive_simulation():
    """
    交互式单帧仿真示例
    """
    radar_sim = RadarSimulator("PD-LS02")
    
    # 创建复杂运动目标
    target_1 = dict(location=(800, 200, 100), speed=(-25, -5, 0), rcs=0.8, phase=0)
    target_2 = dict(location=(1500, -100, 200), speed=(-15, 0, 0), rcs=0.3, phase=0)
    targets = [target_1, target_2]
    
    radar_sim.clear_results()
    current_targets = targets.copy()
    
    frame_idx = 0
    while True:
        user_input = input(f"帧 {frame_idx + 1} - 按回车继续或输入 'q' 退出: ")
        
        if user_input.lower() == 'q':
            break
        
        # 执行单帧
        result = radar_sim.step(current_targets, time_step=0.3, collect_results=True)
        current_targets = result['updated_targets']
        
        # 显示当前帧结果
        peaks = result['peaks']
        print(f"检测结果: {len(peaks)} 个目标")
        for i, peak in enumerate(peaks):
            print(f"  目标 {i+1}: 距离={peak['range']:6.1f}m, 速度={peak['velocity']:6.1f}m/s")
        
        frame_idx += 1
    
    # 绘制最终航迹图
    if frame_idx > 0:
        radar_sim.plot_trajectory_from_collected(show=True,save_path="interactive_simulation_trajectory.png",)
        print(f"共仿真了 {frame_idx} 帧")

def real_time_visualization():
    """
    实时可视化示例（每帧都显示RD图）
    """
    radar_sim = RadarSimulator("PD-LS02")
    
    target_1 = dict(location=(1000, -50, 100), speed=(-20, -10, 0), rcs=0.5, phase=0)
    target_2 = dict(location=(1200, 200, 200), speed=(40, 10, 0), rcs=0.1, phase=0)
    targets = [target_1, target_2]
    
    radar_sim.clear_results()
    current_targets = targets.copy()
    
    for frame in range(5):
        print(f"处理第 {frame + 1} 帧...")
        
        result = radar_sim.step(current_targets, time_step=0.2, collect_results=True)
        current_targets = result['updated_targets']
        
        # 实时显示当前帧的RD图
        radar_sim.plot_rd_map(
            rd_map=result['rd_map'],
            title=f"Frame {frame + 1} - Range-Doppler Map",
            save_path=f"real_time_visualization_{frame + 1}.png",
            show=True
        )
    
    # 最后显示航迹对比图
    radar_sim.plot_trajectory_from_collected(save_path="real_time_visualization_trajectory.png",
                                             show=True)  
    
def generate_rd_animation():
    # 1. 初始化雷达仿真器
    radar_sim = RadarSimulator("PD-LS02")
    
    # 2. 创建目标场景
    targets = [
        dict(location=(1000, -50, 100), speed=(-20, -20, 0), rcs=0.5),
        dict(location=(2000, 50, 200), speed=(40, 20, 0), rcs=0.1)
    ]
    
    # 3. 运行多帧仿真
    results = radar_sim.simulate_multiple_frames(
        initial_targets=targets,
        frame_count=20,
        time_step=0.3
    )
    
    # 4. 生成动画
    radar_sim.generate_rd_animation(
        results=results,
        output_path="rd_animation.mp4",  # 或 .gif
        fps=5,
        dpi=150,
        cmap="viridis"
    )    

        
def main():
    # 创建雷达仿真器
    radar_sim = RadarSimulator("PD-LS02")

    # 打印雷达参数
    params = radar_sim.get_current_radar_params()
    print("Radar Parameters:")
    for key, value in params.items():
        if isinstance(value, (int, float)):
            if key in ['bandwidth', 'sampling_rate']:
                print(f"  {key}: {value/1e6:.1f} MHz")
            elif key == 'prf':
                print(f"  {key}: {value/1e3:.1f} kHz")
            elif key == 'frequency':
                if isinstance(value, list):
                    print(f"  {key}: {[f/1e9 for f in value]} GHz")
                else:
                    print(f"  {key}: {value/1e9:.1f} GHz")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    # 创建目标
    target_1 = dict(location=(1000, -50, 100),
                    speed=(-20, 10, 0), rcs=0.5, phase=0)
    target_2 = dict(location=(2000, -100,200), 
                    speed=(40, 15, 0), rcs=0.1, phase=0)
    targets = [target_1, target_2]
    
    
    # 添加加速度（可选）
    targets_with_accel = radar_sim.add_acceleration_to_targets(targets, acceleration=(0, 0, -2))
    
    # 方法1: 单帧步进
    print("单帧步进示例:")
    current_targets = targets_with_accel.copy()
    frame_count = 20
    for frame in range(frame_count):
        print(f"\n帧 {frame + 1}:")
        result = radar_sim.step(current_targets, time_step=1.0)
        # 显示检测结果
        peaks = result['peaks']
        print(f"检测到 {len(peaks)} 个目标")
        for i, peak in enumerate(peaks[:5]):
            print(f"  目标 {i+1}: 距离={peak['range']}m, 速度={peak['velocity']}m/s")
            
        # 绘制RD图
        radar_sim.plot_rd_map(
            rd_map=result['rd_map'],
            title="Optimized Radar Range-Doppler Map",
            cmap="jet",
            save_path=f"optimized_rd_map_{frame + 1}.png",
            show=True
        )  
        
        # 绘制距离-脉冲序号图
        radar_sim.plot_range_pulse_map(
            baseband=result['baseband'],
            title="Range-Pulse Map",
            cmap="jet",
            save_path=f"range_pulse_map_{frame + 1}.png",
            show=True
        )                  
        
        # 更新目标用于下一帧
        current_targets = result['updated_targets']   
        
    print("\n仿真完成，开始绘制航迹对比图...")        
    # 方法1: 使用收集的结果直接绘制
    radar_sim.plot_trajectory_from_collected(
        save_path="single_step_trajectory.png",
        show=True
    )
    
    # 方法2: 手动获取结果并绘制
    collected_results = radar_sim.get_collected_results()
    radar_sim.plot_trajectory_comparison(
        results=collected_results,
        save_path="manual_trajectory.png",
        show=True
    )
    
    # 绘制2D空间轨迹
    radar_sim.plot_2d_trajectory(
        results=collected_results,
        save_path="2d_trajectory_single_step.png",
        show=True
    )     

    # # 模拟雷达信号
    # baseband = radar_sim.simulate(targets)

    # # 处理信号
    # rd_map = radar_sim.process_signals(baseband)
    # # 使用不同的窗函数进行比较
    # radar_sim.compare_window_functions(baseband)

    # # 使用替代方法检测目标
    # peaks = radar_sim.find_peaks(rd_map)
    # pprint(peaks, indent=2, width=40, depth=4)

    # # 绘制RD图
    # radar_sim.plot_rd_map(
    #     rd_map=rd_map,
    #     title="Optimized Radar Range-Doppler Map",
    #     cmap="jet",
    #     save_path="optimized_rd_map.png",
    #     show=True
    # )

    # radar_sim.plot_3d_rd_map(rd_map, title="3D Range-Doppler Surface", save_path="3d_rd_surface.png")
    # radar_sim.plot_3d_rd_map_contour(rd_map, title="3D Range-Doppler Contour", save_path="3d_rd_contour.png")


if __name__ == "__main__":
    # 选择不同的运行模式
    print("选择运行模式:")
    print("1. 基本单帧仿真")
    print("2. 交互式单步仿真")
    print("3. 实时可视化")
    print("4. 生成仿真动画")

    try:
        choice = input("输入选择 (1-4): ")    
        if choice == "1":
            main()
        elif choice == "2":
            interactive_simulation()
        elif choice == "3":
            real_time_visualization()
        elif choice == "4":
            generate_rd_animation()
        else:
            print("无效选择，运行基本模式")
            main()
    except KeyboardInterrupt:
        pass        
