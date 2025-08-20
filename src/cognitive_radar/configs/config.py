import os
import re
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass, field, asdict
from .loader import ScientificFloatLoader

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class RadarConfig:
    """雷达配置数据类"""
    radar_id: str
    radar_type: str
    frequency: List[float] = field(default_factory=list)
    pulse_width: float = 0.0
    tx_power: float = 0.0
    pulses: int = 0
    prp: float = 0.0
    fs: float = 0.0
    rf_gain: float = 0.0
    baseband_gain: float = 0.0
    max_unambiguous_range: float = 0.0
    max_unambiguous_speed: float = 0.0
    location: List[float] = field(default_factory=list)
    speed: List[float] = field(default_factory=list)
    description: str = ""
    
    @property
    def center_frequency(self) -> float:
        """计算中心频率"""
        if len(self.frequency) == 2:
            return (self.frequency[0] + self.frequency[1]) / 2
        elif len(self.frequency) == 1:
            return self.frequency[0]
        return 0.0
    
    @property
    def bandwidth(self) -> float:
        """计算带宽"""
        if len(self.frequency) == 2:
            return abs(self.frequency[1] - self.frequency[0])
        
        return 0.0
    
    @property
    def prf(self) -> float:
        """计算脉冲重复频率"""
        return 1.0 / self.prp if self.prp > 0 else 0.0
    
    @property
    def wavelength(self) -> float:
        """计算波长"""
        freq = self.center_frequency
        return 3e8 / freq if freq > 0 else 0.0
    
    @property
    def range_resolution(self) -> float:
        """计算距离分辨率"""
        return 3e8 / (2 * self.bandwidth) if self.bandwidth > 0 else 0.0
    
    @property
    def velocity_resolution(self) -> float:
        """计算速度分辨率"""
        if self.wavelength > 0 and self.pulses > 0:
            return self.wavelength / (2 * self.pulses * self.prp)
        return 0.0
    
    @property
    def max_detectable_rcs(self) -> float:
        """估算最大可探测RCS (dBsm)"""
        # 简化的雷达方程估算
        # RCS_max = (P_t * G^2 * λ^2 * σ) / ((4π)^3 * R^4 * SNR_min * kTBF)
        # 这里使用简化模型：RCS_max ∝ P_t * G^2 * λ^2 / R^4
        gain = self.rf_gain + self.baseband_gain
        return 10 * np.log10(
            self.tx_power * (10**(gain/10))**2 * self.wavelength**2 / 
            (self.max_unambiguous_range**4)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def __str__(self) -> str:
        """雷达配置的字符串表示"""
        return (f"RadarConfig({self.radar_id}: {self.radar_type}, "
                f"Freq: {self.center_frequency/1e9:.2f}GHz, "
                f"Range: {self.max_unambiguous_range/1000:.1f}km, "
                f"Speed: {self.max_unambiguous_speed:.1f}m/s)")


class RadarConfigLoader:
    """雷达配置加载器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化雷达配置加载器
        
        :param config_path: 配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            # 使用默认路径
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            self.config_path = base_dir / "assets" / "configs" / "radar" / "radar_config.yml"
        else:
            self.config_path = Path(config_path)
        
        # 加载配置
        self.radar_configs: Dict[str, RadarConfig] = {}
        self._load_config()
    
    def _load_config(self):
        """从YAML文件加载雷达配置"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"雷达配置文件不存在: {self.config_path}")
        
        logger.info(f"加载雷达配置文件: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                # 使用自定义的ScientificFloatLoader处理科学计数法
                config_data = yaml.load(f, Loader=ScientificFloatLoader)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
        
        # 解析雷达配置
        radars_data = config_data.get('radars', {})
        if not radars_data:
            logger.warning("配置文件中未找到雷达配置")
            return
        
        for radar_id, radar_data in radars_data.items():
            try:
                # 处理可能的键名变化
                frequency = radar_data.get('frequency') or radar_data.get('frequency_range')
                pulse_width = radar_data.get('pulse_width') or radar_data.get('pulse_duration')
                prp = radar_data.get('prp') or radar_data.get('pulse_repetition_period')
                pulses = radar_data.get('pulses') or radar_data.get('pulse_count')
                fs = radar_data.get('fs') or radar_data.get('sampling_rate')
                
                # 确保数值类型正确
                frequency = self._ensure_list_float(frequency)
                pulse_width = float(pulse_width) if pulse_width is not None else 0.0
                prp = float(prp) if prp is not None else 0.0
                pulses = int(pulses) if pulses is not None else 0
                fs = float(fs) if fs is not None else 0.0
                
                # 创建雷达配置对象
                config = RadarConfig(
                    radar_id=radar_id,
                    radar_type=radar_data.get('radar_type', ''),
                    frequency=frequency,
                    pulse_width=pulse_width,
                    tx_power=float(radar_data.get('tx_power', 0)),
                    pulses=pulses,
                    prp=prp,
                    fs=fs,
                    rf_gain=float(radar_data.get('rf_gain', 0)),
                    baseband_gain=float(radar_data.get('baseband_gain', 0)),
                    max_unambiguous_range=float(radar_data.get('max_unambiguous_range', 0)),
                    max_unambiguous_speed=float(radar_data.get('max_unambiguous_speed', 0)),
                    location=self._ensure_list_float(radar_data.get('location', [0, 0, 0])),
                    speed=self._ensure_list_float(radar_data.get('speed', [0, 0, 0])),
                    description=radar_data.get('description', '')
                )
                
                self.radar_configs[radar_id] = config
                logger.debug(f"成功加载雷达配置: {radar_id}")
            except Exception as e:
                logger.error(f"解析雷达配置 {radar_id} 失败: {e}")
    
    def _ensure_list_float(self, value: Union[list, float, str]) -> List[float]:
        """确保值转换为浮点数列表"""
        if isinstance(value, list):
            return [float(item) for item in value]
        elif isinstance(value, (float, int)):
            return [float(value)]
        elif isinstance(value, str):
            # 处理科学计数法字符串
            try:
                return [float(value)]
            except ValueError:
                # 尝试解析可能的多值字符串
                values = value.split(',')
                return [float(v.strip()) for v in values]
        return []
    
    def get_radar(self, radar_id: str) -> RadarConfig:
        """获取指定雷达的配置"""
        config = self.radar_configs.get(radar_id)
        if config is None:
            raise KeyError(f"找不到雷达配置: {radar_id}")
        return config
    
    def get_all_radars(self) -> List[RadarConfig]:
        """获取所有雷达配置"""
        return list(self.radar_configs.values())
    
    def get_radar_ids(self) -> List[str]:
        """获取所有雷达ID"""
        return list(self.radar_configs.keys())
    
    def get_radar_by_type(self, radar_type: str) -> List[RadarConfig]:
        """按雷达类型获取配置"""
        return [config for config in self.radar_configs.values() 
                if config.radar_type == radar_type]
    
    def get_radar_by_frequency_band(self, band: str) -> List[RadarConfig]:
        """按频率波段获取雷达配置"""
        band_freqs = {
            'L': (1e9, 2e9),
            'S': (2e9, 4e9),
            'C': (4e9, 8e9),
            'X': (8e9, 12e9),
            'Ku': (12e9, 18e9),
            'K': (18e9, 27e9),
            'Ka': (27e9, 40e9),
            'V': (40e9, 75e9),
            'W': (75e9, 110e9)
        }
        
        if band not in band_freqs:
            raise ValueError(f"未知频率波段: {band}")
        
        min_freq, max_freq = band_freqs[band]
        return [config for config in self.radar_configs.values() 
                if min_freq <= config.center_frequency <= max_freq]
 
    
    def __getitem__(self, radar_id: str) -> RadarConfig:
        """通过索引访问雷达配置"""
        return self.get_radar(radar_id)
    
    def __contains__(self, radar_id: str) -> bool:
        """检查雷达ID是否存在"""
        return radar_id in self.radar_configs
    
    def __iter__(self):
        """迭代雷达配置"""
        return iter(self.radar_configs.values())
    
    def __len__(self) -> int:
        """获取雷达配置数量"""
        return len(self.radar_configs)


# 全局配置加载器实例
_global_config_loader: Optional[RadarConfigLoader] = None

def get_radar_config_loader(config_path: Optional[str] = None) -> RadarConfigLoader:
    """获取全局雷达配置加载器（单例模式）"""
    global _global_config_loader
    
    if _global_config_loader is None:
        _global_config_loader = RadarConfigLoader(config_path)
    
    return _global_config_loader

def get_radar_config(radar_id: str) -> RadarConfig:
    """获取指定雷达的配置"""
    return get_radar_config_loader().get_radar(radar_id)

