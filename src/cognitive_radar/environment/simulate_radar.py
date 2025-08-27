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

    def reset_radar(self):
        self.radar = self.default_radar
        self.last_simulation = None
        self.last_obs = None

    def update_radar(self, params: Dict = {}):
        self.radar = None
        # 根据参数重建雷达
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

        features = self.extract_features(rd_map)

        self.last_obs = {
            'raw_data': baseband,
            'rd_map': rd_map,
            'features': features
        }

        return self.last_obs

    def extract_features(self, rd_map: np.ndarray) -> np.ndarray:
        """Extract features from processed radar data"""
        # Peak detection
        max_val = np.max(rd_map)
        max_idx = np.unravel_index(np.argmax(rd_map), rd_map.shape)

        # Statistical features
        mean_val = np.mean(rd_map)
        std_val = np.std(rd_map)
        energy = np.sum(rd_map**2)

        # Number of detections above threshold
        threshold = mean_val + 2 * std_val
        detections = np.sum(rd_map > threshold)

        return np.array([max_val, *max_idx, mean_val, std_val, energy, detections])

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
            print(f"RD map saved to {save_path}")

        # 显示图像
        if show:
            plt.show()

        return fig

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
    target_1 = dict(location=(1000, 0, 100),
                    speed=(-20, 0, 0), rcs=0.5, phase=0)
    target_2 = dict(location=(2000, 0,200), 
                    speed=(40, 0, 0), rcs=0.1, phase=0)
    targets = [target_1, target_2]

    # 模拟雷达信号
    baseband = radar_sim.simulate(targets)

    # 处理信号
    rd_map = radar_sim.process_signals(baseband)
    # 使用不同的窗函数进行比较
    radar_sim.compare_window_functions(baseband)

    # 使用替代方法检测目标
    peaks = radar_sim.find_peaks(rd_map)
    pprint(peaks, indent=2, width=40, depth=4)

    # 绘制RD图
    radar_sim.plot_rd_map(
        rd_map=rd_map,
        title="Optimized Radar Range-Doppler Map",
        cmap="jet",
        save_path="optimized_rd_map.png",
        show=True
    )

    radar_sim.plot_3d_rd_map(rd_map, title="3D Range-Doppler Surface", save_path="3d_rd_surface.png")
    radar_sim.plot_3d_rd_map_contour(rd_map, title="3D Range-Doppler Contour", save_path="3d_rd_contour.png")


if __name__ == "__main__":
    main()
