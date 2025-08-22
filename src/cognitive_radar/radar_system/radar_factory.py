import numpy as np
from typing import Dict, List, Tuple, Optional
from radarsimpy import Radar, Transmitter, Receiver
from ..configs import get_radar_config_loader, get_radar_config
from .base import RadarFactory

class DefaultRadarFactory(RadarFactory):
    def __init__(self, config_path: Optional[str] = None):
        """
        雷达工厂初始化
        :param config_path: 雷达配置文件路径，可选
        """
        # 获取全局配置加载器
        self.config_loader = get_radar_config_loader(config_path)
        self.radar_types: Dict = self.load_config()
    
    def load_config(self) -> Dict:
        """加载雷达配置信息"""
        return {radar_id: self.config_loader.get_radar(radar_id) 
                for radar_id in self.config_loader.get_radar_ids()}    
   
    def list_available_radars(self) -> List[str]:
        """获取所有可用的雷达类型"""
        return list(self.radar_types.keys())
    
    def get_radar_info(self, radar_type: str) -> Dict:
        """获取雷达描述信息"""
        if radar_type not in self.radar_types:
            raise ValueError(f"未找到雷达配置: {radar_type}")
        
        config = self.radar_types[radar_type]
        return {
            'id': radar_type,
            'type': config.radar_type,
            'description': config.description,
            'fs': config.fs,
            'prf': config.prf,
            'pulses': config.pulses,
            'rf_gain': config.rf_gain,
            'baseband_gain': config.baseband_gain,
            'pulse_width': config.pulse_width,
            'tx_power': config.tx_power,
            'frequency_range': f"{config.frequency[0]/1e9:.1f}-{config.frequency[1]/1e9:.1f} GHz",
            'center_frequency': config.center_frequency,
            'bandwidth': config.bandwidth,
            'max_range': f"{config.max_unambiguous_range/1000:.1f} km",
            'max_speed': f"{config.max_unambiguous_speed:.1f} m/s",
            'range_resolution': f"{config.range_resolution:.2f} m",
            'velocity_resolution': f"{config.velocity_resolution:.2f} m/s",
            'location': config.location,
            'speed': config.speed
        }
    
    def create(self, 
                     radar_type: str, 
                     location: Optional[Tuple[float, float, float]] = None,  
                     speed: Optional[Tuple[float, float, float]] = None) -> Radar:
        """
        创建雷达实例
        :param radar_type: 雷达型号 (如 PD-LS05)
        :param location: 可选，雷达位置 (x, y, z) 元组，单位米
        :param speed: 可选，雷达运动速度 (vx, vy, vz) 元组，单位米/秒
        :return: radarsimpy.Radar 实例
        """
        if radar_type not in self.radar_types:
            raise ValueError(f"不支持的雷达类型: {radar_type}")
        
        config = self.radar_types[radar_type]
        
        # 使用传入的位置和速度或默认值
        loc = location if location is not None else tuple(config.location)
        spd = speed if speed is not None else tuple(config.speed)

        # 示例：30 dB增益的方向图
        antenna_gain = 30  # dBi
        az_angle = np.arange(-20, 21, 1)
        az_pattern = 20 * np.log10(np.cos(az_angle / 180 * np.pi) ** 500) + antenna_gain
        el_angle = np.arange(-20, 21, 1)
        el_pattern = 20 * np.log10((np.cos(el_angle / 180 * np.pi)) ** 400) + antenna_gain  
        tx_channel = dict(
            location=loc,
            azimuth_angle=az_angle,
            azimuth_pattern=az_pattern,
            elevation_angle=el_angle,
            elevation_pattern=el_pattern,
            speed=spd
        ) 
        
                  
        # 创建发射机
        tx = Transmitter(           
            f=config.frequency,
            t=config.pulse_width,     # 脉冲宽度列表
            tx_power=config.tx_power,   # 峰值功率(W)
            pulses=config.pulses,
            prp=config.prp,
            channels=[tx_channel],
        )
        
        rx_channel = dict(
            location=loc,
            azimuth_angle=az_angle,
            azimuth_pattern=az_pattern,
            elevation_angle=el_angle,
            elevation_pattern=el_pattern,
            speed=spd
        )         
        
        # 创建接收机
        rx = Receiver(
            fs=config.fs,               # 采样率(Hz)
            noise_figure=6,
            rf_gain=config.rf_gain,
            baseband_gain=config.baseband_gain,
            channels=[rx_channel]
        )
        
        # 创建雷达实例
        return Radar(
            transmitter=tx,
            receiver=rx,
            location=loc,
            speed=spd,
            seed=42
        )
        
def main():
    # 创建雷达工厂
    factory = DefaultRadarFactory()
    
    # 列出所有可用雷达
    print("可用雷达型号:", factory.list_available_radars())
    
    # 获取雷达信息
    for radar_id in factory.list_available_radars():
        info = factory.get_radar_info(radar_id)
        print(f"\n雷达 {radar_id} 信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # 创建特定雷达实例
    radar_type = "PD-LS02"  # 反无人机专用雷达
    radar = factory.create(
        radar_type,
        location=(0, 0, 0),  # 自定义位置
        speed=(0, 0, 0)          # 静止
    )
    
    print(f"\n成功创建 {radar_type} 雷达实例:")
    
    # 正确访问雷达参数的方式：
    # 1. 通过 radar_prop 访问发射机
    transmitter = radar.radar_prop["transmitter"]
    receiver = radar.radar_prop["receiver"]
    
    # 2. 中心频率 - 通过发射机的频率参数
    # print(f"中心频率: {transmitter.waveform_prop} ")
    print(f"中心频率: {np.mean(transmitter.waveform_prop['f'])/1e9:.2f} GHz")
    
    # 3. 带宽 - 通过发射机的带宽属性
    print(f"带宽: {transmitter.waveform_prop['bandwidth']/1e6:.1f} MHz")
    
    # 4. 脉冲宽度 - 通过发射机的脉冲宽度参数
    print(f"脉冲宽度: {transmitter.waveform_prop['pulse_length']*1e6:.2f} μs")
    
    # 5. 位置 - 通过发射机通道的位置属性
    # 注意：我们创建时只有一个通道，所以索引为0
    print(f"位置: {transmitter.txchannel_prop['locations']}")
    
    # 6. 发射通道个数 - 发射机发射通道个数
    print(f"发射通道: {transmitter.txchannel_prop['size']}")
    
    # 7. 天线增益 - 发射机通道的天线增益属性
    print(f"天线增益: {transmitter.txchannel_prop['antenna_gains']}")
    
    # 87. 采样率 - 通过接收机的采样率属性
    print(f"采样率: {receiver.bb_prop['fs']/1e6:.1f} MHz")
    
    # 8. 最大不模糊距离 - 通过配置信息
    config = factory.radar_types[radar_type]
    print(f"最大不模糊距离: {config.max_unambiguous_range/1000:.1f} km")
    
    # 9. 最大不模糊速度 - 通过配置信息
    print(f"最大不模糊速度: {config.max_unambiguous_speed:.1f} m/s")
    
    # 10. 距离分辨率 - 通过配置信息
    print(f"距离分辨率: {config.range_resolution:.2f} m")
    
    # 11. 速度分辨率 - 通过配置信息
    print(f"速度分辨率: {config.velocity_resolution:.2f} m/s")    

if __name__ == "__main__":
    main()