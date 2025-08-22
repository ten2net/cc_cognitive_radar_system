from typing import Dict
import numpy as np
import radarsimpy as rp
from ..radar_system import RadarFactory, DefaultRadarFactory
from radarsimpy.simulator import sim_radar
import radarsimpy.processing as proc
from scipy.constants import speed_of_light
from scipy import signal

def normalize_rd_map(rd_map):
    """归一化距离-多普勒图"""
    # 方法1：按最大值归一化
    max_val = np.max(np.abs(rd_map))
    if max_val > 0:
        return rd_map / max_val
    
    # 方法2：按中值归一化
    median_val = np.median(np.abs(rd_map))
    if median_val > 0:
        return rd_map / median_val
    
    return rd_map

class RadarSimulator:
    """Wrapper for radar simulation using RadarSimPy"""
    
    def __init__(self, radar_type: str, params: Dict = {}):
        self.radar = DefaultRadarFactory().create(radar_type)
        self.default_radar = DefaultRadarFactory().create(radar_type)
        self.last_simulation = None
        self.last_obs = None
        
        # 优化雷达参数
        # self.optimize_radar_parameters()
        
    def optimize_radar_parameters(self):
        """优化雷达参数以获得更好的距离多普勒图"""
        # 设置更合理的参数
        # 1. 增加带宽以提高距离分辨率
        self.radar.radar_prop['transmitter'].waveform_prop["bandwidth"] = 300e6  # 300 MHz带宽
        
        # 2. 调整PRF以平衡距离和速度不模糊范围
        prf = 10e3  # 10 kHz PRF
        self.radar.radar_prop['transmitter'].waveform_prop["prp"] = [1/prf]  # 脉冲重复周期
        
        # 3. 增加脉冲数以提高速度分辨率
        self.radar.radar_prop['transmitter'].waveform_prop["pulses"] = 256
        
        # 4. 调整脉冲宽度
        self.radar.radar_prop['transmitter'].waveform_prop["pulse_length"] = 10e-6  # 10 μs
        
        # 5. 调整采样率以适应新的带宽
        self.radar.radar_prop['receiver'].bb_prop["fs"] = 400e6  # 400 MHz采样率
        self.radar.sample_prop["samples_per_pulse"] = int(400e6 * 10e-6)  # 根据脉冲宽度计算采样点数
        
        # 6. 设置中心频率
        self.radar.radar_prop['transmitter'].waveform_prop["f"] = [77e9]  # 77 GHz
        
        # 7. 调整发射功率
        self.radar.radar_prop['transmitter'].rf_prop["tx_power"] = 12  # 12 dBm
        
        # 8. 设置天线参数
        self.radar.radar_prop['transmitter'].txchannel_prop["az_angles"] = [-30, 0, 30]
        self.radar.radar_prop['transmitter'].txchannel_prop["el_angles"] = [-10, 0, 10]
        
    def simulate(self, targets: list) -> np.ndarray:
        """Run radar simulation with current parameters"""
        data = sim_radar(
            self.radar,
            targets
        )
        
        self.last_simulation = data
        timestamp = data["timestamp"]
        baseband=data["baseband"]
        noise = data["noise"] 
        return baseband + noise 
    
    def process_signals(self, baseband: np.ndarray) -> np.ndarray:
        """Process raw radar signals"""
        
        # 计算每个脉冲的采样点数
        samples_per_pulse = self.radar.sample_prop["samples_per_pulse"]
        pulses = self.radar.radar_prop["transmitter"].waveform_prop["pulses"]
        
        # 创建窗函数
        range_window = signal.windows.chebwin(samples_per_pulse, at=60)
        dop_window = signal.windows.chebwin(pulses, at=60)      
        
        # 进行距离FFT，使用窗函数和指定FFT点数
        self.range_data = proc.range_fft(
            baseband, 
            rwin=range_window)
        
        # 多普勒FFT
        self.doppler_data =proc.doppler_fft(self.range_data, 
                                             dwin=dop_window)       
        
        # 距离-多普勒FFT
        range_fft_points = 4096  # 增加FFT点数
        rd_map = proc.range_doppler_fft(
            baseband, 
            rwin=range_window,
            dwin=dop_window,
            # rn=samples_per_pulse,
            rn=range_fft_points,  # 使用更大的FFT点数
            dn=pulses)  
        
        return rd_map.squeeze(0)
    
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
    
    def update_radar_params(self, params: Dict):
        """Update radar parameters"""
        # 这里可以添加参数更新逻辑
        pass
        
    def reset_radar(self) -> None:
        """Reset radar to default parameters"""
        self.radar = self.default_radar
        self.last_simulation = None
        self.last_obs = None
        
        # 重新优化参数
        # self.optimize_radar_parameters()
        
    def randomize_radar(self) -> None:
        """Randomize radar parameters for domain randomization"""
        self.radar.radar_prop['receiver'].rf_prop["noise_figure"] = np.random.uniform(10, 15)
        self.radar.radar_prop['transmitter'].rf_prop["tx_power"] = np.random.uniform(5, 15)
        
    def get_current_radar_params(self) -> Dict:
        params = {}
        
        # 获取基本参数
        params['beam_az'] = self.radar.radar_prop['transmitter'].txchannel_prop["az_angles"]
        params['beam_el'] = self.radar.radar_prop['transmitter'].txchannel_prop["el_angles"]
        params['gain'] = self.radar.radar_prop['receiver'].rxchannel_prop["antenna_gains"]
        
        params['bandwidth'] = self.radar.radar_prop['transmitter'].waveform_prop["bandwidth"]
        params['frequency'] = self.radar.radar_prop['transmitter'].waveform_prop["f"]
        
        params['pulse_width'] = self.radar.radar_prop['transmitter'].waveform_prop["pulse_length"]
        prp = self.radar.radar_prop['transmitter'].waveform_prop["prp"][0]
        prf = 1 / prp
        params['prf'] = prf
        params['tx_power'] = self.radar.radar_prop['transmitter'].rf_prop["tx_power"]
        
        # 计算波长
        wavelength = speed_of_light / np.mean(params['frequency'])
        pulses = self.radar.radar_prop['transmitter'].waveform_prop["pulses"]
        
        # 正确的速度分辨率计算
        params['velocity_resolution'] = wavelength * prf / (2 * pulses)
        
        # 正确的多普勒分辨率计算
        params['doppler_resolution'] = prf / pulses
        
        # 其他参数
        params['range_resolution'] = speed_of_light / (2 * params['bandwidth'])
        params['max_unambiguous_range'] = speed_of_light / (2 * prf)
        params['max_unambiguous_velocity'] = prf * wavelength / 4
        
        # 添加采样率和采样点数
        params['sampling_rate'] = self.radar.radar_prop['receiver'].bb_prop["fs"]
        params['samples_per_pulse'] = self.radar.sample_prop["samples_per_pulse"]
        
        return params
    
    def plot_rd_map(self, rd_map: np.ndarray = None, title: str = "Range-Doppler Map",
                    figsize: tuple = (12, 8), cmap: str = "jet",
                    save_path: str = None, show: bool = True):
        """
        绘制专业级距离-多普勒图(RD图)
        
        参数:
            rd_map (np.ndarray): 距离-多普勒图数据，形状为(距离单元数, 多普勒单元数)
            title (str): 图表标题
            figsize (tuple): 图表尺寸(宽, 高)
            cmap (str): 颜色映射名称
            save_path (str): 图像保存路径（可选）
            show (bool): 是否显示图像
        """
        # 导入放在方法内部
        import matplotlib.pyplot as plt
        
        if rd_map is None and self.last_obs is not None:
            rd_map = self.last_obs['rd_map']
        elif rd_map is None:
            raise ValueError("No RD map available to plot")
            
        # 获取雷达参数
        params = self.get_current_radar_params()
        range_res = params.get('range_resolution', 1.0)
        
        # 计算波长
        wavelength = speed_of_light / np.mean(params['frequency'])
        prf = params['prf']
        
        # 正确的多普勒速度范围计算
        max_doppler_velocity = prf * wavelength / 4  # 最大不模糊速度
        
        # 获取矩阵尺寸
        num_range_bins, num_doppler_bins = rd_map.shape
        
        # 计算实际物理范围
        max_range = 3000 # params.get('max_unambiguous_range', num_range_bins * range_res)
        
        # 计算幅度(取绝对值)
        magnitude = np.abs(rd_map)
        
        # 创建图表和坐标轴
        fig, ax = plt.subplots(figsize=figsize)
        
        # 使用对数尺度显示(更好地展示动态范围)
        log_magnitude = 10 * np.log10(magnitude + 1e-9)  # 转换为dB尺度
        
        # 创建热图 - 使用物理单位作为范围
        im = ax.imshow(log_magnitude, 
                aspect='auto', 
                cmap=cmap,
                origin='lower',
                extent=[0, max_range, -max_doppler_velocity, max_doppler_velocity])
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Magnitude (dB)', fontsize=12)
        
        # 设置坐标轴标签
        ax.set_xlabel('Range (m)', fontsize=12)
        ax.set_ylabel('Velocity (m/s)', fontsize=12)
        
        # 动态设置距离刻度 - 根据max_range计算
        # 计算合适的刻度数量和间隔
        if max_range <= 500:
            num_ticks = 6  # 小范围使用更多刻度
            tick_step = max_range / (num_ticks - 1)
        elif max_range <= 2000:
            num_ticks = 5  # 中等范围使用5个刻度
            tick_step = max_range / (num_ticks - 1)
        else:
            num_ticks = 5  # 大范围也使用5个刻度
            tick_step = max_range / (num_ticks - 1)
        
        # 生成刻度位置和标签
        x_ticks = [i * tick_step for i in range(num_ticks)]
        x_labels = [str(int(tick)) for tick in x_ticks]
        
        # 设置距离刻度
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        
        # 设置多普勒刻度 - 根据参考图片的分布
        y_ticks = [-max_doppler_velocity, -max_doppler_velocity/2, 0, 
                   max_doppler_velocity/2, max_doppler_velocity]
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
                # 每500米添加一次文本标注
                if distance % 500 == 0 and distance > 0:
                    ax.text(distance + 10, max_doppler_velocity * 0.9, 
                            f'{distance}m', color='white', fontsize=9, weight='bold')
        
        # 添加雷达参数信息
        param_text = (
            f"Bandwidth: {params['bandwidth']/1e6:.1f} MHz\n"
            f"PRF: {params['prf']/1e3:.1f} kHz\n"
            # f"Pulses: {params['pulses']}\n"
            f"Range Res: {params['range_resolution']:.2f} m\n"
            f"Velocity Res: {params['velocity_resolution']:.2f} m/s"
        )
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
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
    target_1 = dict(location=(1000, 0, 100), speed=(-11.2, 0, 0), rcs=0.5, phase=0)
    target_2 = dict(location=(2000, 0, 200), speed=(11 , 0, 0), rcs=0.5, phase=0)
    targets = [target_1, target_2]  

    # 模拟雷达信号
    baseband = radar_sim.simulate(targets)

    # 处理信号
    rd_map = radar_sim.process_signals(baseband)

    # 绘制RD图
    radar_sim.plot_rd_map(
        rd_map=rd_map,
        title="Optimized Radar Range-Doppler Map",
        cmap="jet",
        save_path="optimized_rd_map.png",
        show=True
    )
    
if __name__ == "__main__":
    main()