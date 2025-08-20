import numpy as np
from typing import Dict, List, Tuple, Optional
from .base import RadarFactory
from .radar_factory import DefaultRadarFactory

class LowSlowSmallRadarSystem:
    """低慢小目标探测雷达系统"""
    
    def __init__(self, 
                 radar_type: str,
                 radar_factory: RadarFactory = DefaultRadarFactory(),
                 target_rcs: float = 0.01,  # 目标RCS (m²)
                 noise_figure: float = 6.0,  # 噪声系数 (dB)
                 system_loss: float = 3.0,   # 系统损耗 (dB)
                 detection_threshold: float = 13.0,  # 检测门限 (dB)
                 false_alarm_rate: float = 1e-6,  # 虚警率
                 weather_attenuation: float = 0.005  # 天气衰减 (dB/km)
                 ):
        """
        :param radar_type: 雷达类型
        :param radar_factory: 雷达工厂
        :param target_rcs: 目标雷达截面积 (m²)
        :param noise_figure: 接收机噪声系数 (dB)
        :param system_loss: 系统损耗 (dB)
        :param detection_threshold: 检测门限 (dB)
        :param false_alarm_rate: 虚警率
        :param weather_attenuation: 天气衰减 (dB/km)
        """
        self.radar_factory = radar_factory
        self.radar = radar_factory.create(radar_type)
        self.radar_info = radar_factory.get_radar_info(radar_type)
        
        # 基本雷达参数
        self.freq_range = self.radar_info['frequency_range']
        self.center_freq = self.radar_info['center_frequency']
        self.bandwidth = self.radar_info['bandwidth']
        self.prf = self.radar_info['prf']
        self.num_pulses = self.radar_info['pulses']
        self.pulse_width = self.radar_info['pulse_width']
        self.tx_power = self.radar_info['tx_power']
        self.wavelength = 3e8 / self.center_freq
        
        # 系统参数
        self.target_rcs = target_rcs
        self.noise_figure = noise_figure
        self.system_loss = system_loss
        self.detection_threshold = detection_threshold
        self.false_alarm_rate = false_alarm_rate
        self.weather_attenuation = weather_attenuation
        
        # 计算雷达性能参数
        self._calculate_performance_metrics()
    
    def _calculate_performance_metrics(self):
        """计算雷达性能指标"""
        # 1. 基本性能
        self.range_resolution = 3e8 / (2 * self.bandwidth)
        self.max_unambiguous_range = 3e8 / (2 * self.prf)
        self.max_unambiguous_velocity = self.prf * self.wavelength / 4
        
        # 2. 速度分辨率
        self.velocity_resolution = self.wavelength / (2 * self.num_pulses * (1/self.prf))
        
        # 3. 多普勒分辨率
        self.doppler_resolution = 2 * self.velocity_resolution / self.wavelength
        
        # 4. 噪声功率
        k = 1.38e-23  # 玻尔兹曼常数
        T0 = 290      # 标准温度 (K)
        self.noise_power = k * T0 * self.bandwidth * 10**(self.noise_figure/10)
        
        # 5. 最小可检测信号 (MDS)
        self.min_detectable_signal = self.noise_power * 10**(self.detection_threshold/10)
        
        # 6. 最大探测距离 (基于雷达方程)
        gain = self.radar_info['rf_gain'] + self.radar_info['baseband_gain']
        self.max_detection_range = self._calculate_max_range(
            self.tx_power, gain, self.target_rcs, self.min_detectable_signal
        )
        
        # 7. 检测概率计算参数
        self._calculate_detection_probability_params()
    
    def _calculate_max_range(self, tx_power: float, gain: float, rcs: float, min_signal: float) -> float:
        """计算最大探测距离 (雷达方程)"""
        # 雷达方程: R_max = [P_t * G^2 * λ^2 * σ / ( (4π)^3 * S_min * L ) ]^(1/4)
        numerator = tx_power * (10**(gain/10))**2 * self.wavelength**2 * rcs
        denominator = (4 * np.pi)**3 * min_signal * 10**(self.system_loss/10)
        return (numerator / denominator) ** 0.25
    
    def _calculate_detection_probability_params(self):
        """计算检测概率相关参数"""
        # 虚警概率参数
        self.false_alarm_probability = self.false_alarm_rate * self.bandwidth
        
        # 信噪比门限 (基于虚警概率)
        # 使用近似公式: SNR_threshold ≈ ln(1/P_fa)
        if self.false_alarm_probability <= 0:
            self.false_alarm_probability = 1e-10
        self.snr_threshold = np.log(1 / self.false_alarm_probability)
        
        # 检测概率计算 (基于Swerling模型)
        # 这里使用Swerling I模型 (慢起伏目标)
        self.detection_probability = lambda snr: 1 - (1 + self.snr_threshold/snr)**(-1)
    
    def calculate_snr_at_range(self, distance: float) -> float:
        """
        计算给定距离处的信噪比
        :param distance: 目标距离 (m)
        :return: 信噪比 (dB)
        """
        # 雷达方程: SNR = (P_t * G^2 * λ^2 * σ) / ( (4π)^3 * k * T * B * F * L * R^4 )
        gain = self.radar_info['rf_gain'] + self.radar_info['baseband_gain']
        numerator = (self.tx_power * (10**(gain/10))**2 * self.wavelength**2 * self.target_rcs)
        denominator = ((4 * np.pi)**3 * self.noise_power * 10**(self.system_loss/10) * distance**4)
        
        # 考虑天气衰减
        weather_loss = np.exp(-0.2 * self.weather_attenuation * distance / 1000)
        snr_linear = (numerator / denominator) * weather_loss
        
        # 避免负值
        if snr_linear <= 0:
            snr_linear = 1e-10
            
        return 10 * np.log10(snr_linear)
    
    def calculate_detection_probability(self, distance: float) -> float:
        """
        计算给定距离处的检测概率
        :param distance: 目标距离 (m)
        :return: 检测概率 (0-1)
        """
        snr = self.calculate_snr_at_range(distance)
        snr_linear = 10**(snr / 10)  # 转换为线性值
        return self.detection_probability(snr_linear)
    
    def calculate_range_accuracy(self, snr: float) -> float:
        """
        计算距离测量精度
        :param snr: 信噪比 (dB)
        :return: 距离精度 (m)
        """
        # 距离精度 ≈ (c / (2 * B * sqrt(2 * SNR)))
        snr_linear = 10**(snr / 10)
        return 3e8 / (2 * self.bandwidth * np.sqrt(2 * snr_linear))
    
    def calculate_velocity_accuracy(self, snr: float) -> float:
        """
        计算速度测量精度
        :param snr: 信噪比 (dB)
        :return: 速度精度 (m/s)
        """
        # 速度精度 ≈ (λ / (2 * T_obs * sqrt(2 * SNR)))
        snr_linear = 10**(snr / 10)
        T_obs = self.num_pulses / self.prf  # 观测时间
        return self.wavelength / (2 * T_obs * np.sqrt(2 * snr_linear))
    
    def calculate_beam_coverage(self, distance: float) -> Tuple[float, float]:
        """
        计算波束覆盖范围
        :param distance: 目标距离 (m)
        :return: (方位覆盖范围, 俯仰覆盖范围) (m)
        """
        # 假设波束宽度为3度 (简化模型)
        beamwidth_az = np.deg2rad(3)  # 方位波束宽度 (弧度)
        beamwidth_el = np.deg2rad(3)  # 俯仰波束宽度 (弧度)
        
        az_coverage = distance * beamwidth_az
        el_coverage = distance * beamwidth_el
        return (az_coverage, el_coverage)
    
    def calculate_pulse_compression_gain(self) -> float:
        """
        计算脉冲压缩增益
        :return: 脉冲压缩增益 (dB)
        """
        # 增益 = 10 * log10(时间带宽积)
        time_bandwidth_product = self.pulse_width * self.bandwidth
        return 10 * np.log10(time_bandwidth_product)
    
    def calculate_clutter_rcs(self, clutter_type: str, area: float) -> float:
        """
        计算杂波RCS
        :param clutter_type: 杂波类型 ('ground', 'sea', 'urban')
        :param area: 杂波区域面积 (m²)
        :return: 杂波RCS (m²)
        """
        # 杂波反射率 (σ⁰) 典型值 (dB/m²)
        clutter_reflectivity = {
            'ground': -20,  # 草地
            'sea': -30,     # 海面
            'urban': -10    # 城市
        }.get(clutter_type, -20)
        
        # 转换为线性值并计算总RCS
        sigma0 = 10**(clutter_reflectivity / 10)
        return sigma0 * area
    
    def calculate_clutter_to_noise_ratio(self, distance: float, clutter_rcs: float) -> float:
        """
        计算杂噪比 (CNR)
        :param distance: 杂波距离 (m)
        :param clutter_rcs: 杂波RCS (m²)
        :return: 杂噪比 (dB)
        """
        # 计算杂波信号功率
        gain = self.radar_info['rf_gain'] + self.radar_info['baseband_gain']
        clutter_power = (self.tx_power * (10**(gain/10))**2 * self.wavelength**2 * clutter_rcs) / \
                        ((4 * np.pi)**3 * distance**4 * 10**(self.system_loss/10))
        
        # 计算CNR
        cnr_linear = clutter_power / self.noise_power
        
        # 避免负值
        if cnr_linear <= 0:
            cnr_linear = 1e-10
            
        return 10 * np.log10(cnr_linear)
    
    def calculate_mti_improvement_factor(self, num_pulses: Optional[int] = None) -> float:
        """
        计算动目标显示(MTI)改善因子
        :param num_pulses: 使用的脉冲数 (可选)
        :return: 改善因子 (dB)
        """
        # 简化模型: 改善因子 ≈ 10 * log10(N) + 6 dB (对于双延迟线对消器)
        N = num_pulses if num_pulses is not None else self.num_pulses
        
        # 确保N大于0
        if N <= 0:
            N = 1
            
        return 10 * np.log10(N) + 6
    
    def calculate_required_snr_for_pd(self, pd: float, pfa: Optional[float] = None) -> float:
        """
        计算达到指定检测概率所需的最小SNR
        :param pd: 期望检测概率 (0-1)
        :param pfa: 虚警概率 (可选，默认使用系统设置)
        :return: 所需最小SNR (dB)
        """
        # 使用Swerling I模型近似
        pfa_val = pfa if pfa is not None else self.false_alarm_probability
        
        # 避免除以零和对数域错误
        if pd >= 1.0:
            pd = 0.9999
        if pfa_val <= 0:
            pfa_val = 1e-10
        
        # 计算所需SNR (线性值)
        snr_linear = np.log(1 / (1 - pd)) / np.log(1 / pfa_val) - 1
        
        # 避免负值
        if snr_linear <= 0:
            snr_linear = 1e-10
            
        return 10 * np.log10(snr_linear)
    
    def calculate_max_detection_range_for_pd(self, pd: float) -> float:
        """
        计算达到指定检测概率的最大探测距离
        :param pd: 期望检测概率 (0-1)
        :return: 最大探测距离 (m)
        """
        # 计算所需SNR
        required_snr = self.calculate_required_snr_for_pd(pd)
        required_snr_linear = 10**(required_snr / 10)
        
        # 雷达方程反推距离
        gain = self.radar_info['rf_gain'] + self.radar_info['baseband_gain']
        numerator = (self.tx_power * (10**(gain/10))**2 * self.wavelength**2 * self.target_rcs)
        denominator = ((4 * np.pi)**3 * self.noise_power * 10**(self.system_loss/10) * required_snr_linear)
        
        # 避免分母为零
        if denominator <= 0:
            denominator = 1e-10
            
        return (numerator / denominator) ** 0.25
    
    def generate_range_profile(self, targets: List[Dict]) -> np.ndarray:
        """
        生成距离剖面 (简化模型)
        :param targets: 目标列表 [{'distance':, 'rcs':, 'velocity':}]
        :return: 距离剖面 (dB)
        """
        # 创建距离门
        range_bins = np.linspace(0, self.max_unambiguous_range, 512)
        profile = np.zeros_like(range_bins)
        
        # 添加每个目标
        for target in targets:
            # 计算目标在距离门中的位置
            bin_idx = int(target['distance'] / (self.max_unambiguous_range / len(range_bins)))
            
            # 计算目标信号强度 (简化模型)
            snr = self.calculate_snr_at_range(target['distance'])
            signal_strength = 10**(snr / 10)
            
            # 添加到剖面
            if bin_idx < len(profile):
                profile[bin_idx] += signal_strength
        
        # 添加噪声
        noise = np.random.normal(0, np.sqrt(self.noise_power), len(profile))
        profile += noise
        
        # 转换为dB
        return 10 * np.log10(np.abs(profile) + 1e-10)  # 避免log(0)
    
    def print_system_summary(self):
        """打印雷达系统性能摘要"""
        print("\n=== 低慢小目标雷达系统性能摘要 ===")
        print(f"雷达型号: {self.radar_info['id']} ({self.radar_info['type']})")
        print(f"中心频率: {self.center_freq/1e9:.2f} GHz")
        print(f"带宽: {self.bandwidth/1e6:.1f} MHz")
        print(f"PRF: {self.prf/1e3:.1f} kHz")
        print(f"脉冲宽度: {self.pulse_width*1e6:.2f} μs")
        print(f"发射功率: {self.tx_power/1e3:.1f} kW")
        print(f"脉冲数: {self.num_pulses}")
        print("\n性能参数:")
        print(f"距离分辨率: {self.range_resolution:.2f} m")
        print(f"速度分辨率: {self.velocity_resolution:.2f} m/s")
        print(f"最大无模糊距离: {self.max_unambiguous_range/1000:.1f} km")
        print(f"最大无模糊速度: {self.max_unambiguous_velocity:.1f} m/s")
        print(f"最大探测距离 (RCS={self.target_rcs}m²): {self.max_detection_range/1000:.1f} km")
        print(f"脉冲压缩增益: {self.calculate_pulse_compression_gain():.1f} dB")
        print(f"MTI改善因子: {self.calculate_mti_improvement_factor():.1f} dB")
        print(f"最小可检测信号: {10*np.log10(self.min_detectable_signal*1000):.1f} dBm")
        print("="*40)
        
# 测试修复后的函数
def main():
    # 创建雷达系统
    radar_system = LowSlowSmallRadarSystem(
        radar_type="PD-LS01",
        target_rcs=0.1,
        noise_figure=5.0,
        system_loss=4.0,
        detection_threshold=15.0,
        false_alarm_rate=1e-6,
        weather_attenuation=0.01
    )
    
    # 测试MTI改善因子计算
    print("MTI改善因子测试:")
    print(f"默认脉冲数: {radar_system.calculate_mti_improvement_factor():.1f} dB")
    print(f"指定脉冲数(10): {radar_system.calculate_mti_improvement_factor(10):.1f} dB")
    print(f"指定脉冲数(0): {radar_system.calculate_mti_improvement_factor(0):.1f} dB")
    
    # 测试所需SNR计算
    print("\n所需SNR测试:")
    print(f"PD=0.9: {radar_system.calculate_required_snr_for_pd(0.9):.1f} dB")
    print(f"PD=1.0: {radar_system.calculate_required_snr_for_pd(1.0):.1f} dB")
    print(f"PD=0.0: {radar_system.calculate_required_snr_for_pd(0.0):.1f} dB")
    
    # 测试边界情况
    print("\n边界情况测试:")
    print(f"PD=0.9999: {radar_system.calculate_required_snr_for_pd(0.9999):.1f} dB")
    print(f"PFA=0: {radar_system.calculate_required_snr_for_pd(0.9, 0):.1f} dB")
    
    # 测试最大探测距离计算
    print("\n最大探测距离测试:")
    print(f"PD=0.9: {radar_system.calculate_max_detection_range_for_pd(0.9)/1000:.1f} km")
    print(f"PD=1.0: {radar_system.calculate_max_detection_range_for_pd(1.0)/1000:.1f} km")
    
    radar_system.print_system_summary()

if __name__ == "__main__":  
    main()        