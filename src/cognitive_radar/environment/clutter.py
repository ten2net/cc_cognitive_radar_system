import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
from .scene import SceneModel

@dataclass
class ClutterModel:
    """杂波模型基类"""
    scene: SceneModel
    
    def get_clutter_rcs(self, position: Tuple[float, float], frequency: float) -> float:
        """获取指定位置的杂波RCS (m²)"""
        raise NotImplementedError("子类必须实现此方法")
    
    def generate_clutter_map(self, bounds: Tuple[float, float, float, float], 
                            resolution: float, frequency: float) -> np.ndarray:
        """生成杂波RCS地图"""
        min_x, min_y, max_x, max_y = bounds
        cols = int((max_x - min_x) / resolution)
        rows = int((max_y - min_y) / resolution)
        
        clutter_map = np.zeros((rows, cols))
        
        for i in range(rows):
            y = min_y + i * resolution
            for j in range(cols):
                x = min_x + j * resolution
                clutter_map[i, j] = self.get_clutter_rcs((x, y), frequency)
        
        return clutter_map

class ConstantClutterModel(ClutterModel):
    """常数杂波模型"""
    
    def __init__(self, scene: SceneModel, rcs_density: float = 0.1):
        """
        :param rcs_density: 杂波RCS密度 (m²/m²)
        """
        super().__init__(scene)
        self.rcs_density = rcs_density
    
    def get_clutter_rcs(self, position: Tuple[float, float], frequency: float) -> float:
        """获取指定位置的杂波RCS (m²)"""
        x, y = position
        
        # 检查是否在建筑物内
        building = self.scene.get_building_at_position(x, y)
        if building:
            return building.rcs  # 建筑物有固定RCS
        
        # 获取地表类型
        terrain_type = self.scene.terrain.terrain_type
        
        # 不同地表类型的RCS密度
        density_factors = {
            "urban": 0.5,
            "forest": 0.3,
            "grass": 0.1,
            "water": 0.05,
            "desert": 0.02
        }
        
        factor = density_factors.get(terrain_type, 0.1)
        return self.rcs_density * factor

class EmpiricalClutterModel(ClutterModel):
    """经验杂波模型 (基于Billingsley模型)"""
    
    def __init__(self, scene: SceneModel):
        super().__init__(scene)
        # 不同地表类型的杂波参数 (A, B, C)
        self.clutter_params = {
            "urban": (0.32, 0.39, 0.17),
            "forest": (0.25, 0.35, 0.15),
            "grass": (0.15, 0.30, 0.10),
            "farmland": (0.10, 0.25, 0.08),
            "water": (0.05, 0.20, 0.05),
            "desert": (0.02, 0.15, 0.03)
        }
    
    def get_clutter_rcs(self, position: Tuple[float, float], frequency: float) -> float:
        """获取指定位置的杂波RCS (dBsm/m²)"""
        x, y = position
        
        # 检查是否在建筑物内
        building = self.scene.get_building_at_position(x, y)
        if building:
            # 建筑物使用固定RCS
            return building.rcs
        
        # 获取地表类型
        terrain_type = self.scene.terrain.terrain_type
        
        # 获取杂波参数
        A, B, C = self.clutter_params.get(terrain_type, (0.15, 0.30, 0.10))
        
        # 计算杂波RCS密度 (dBsm/m²)
        # σ⁰ = A + B * log10(f) + C * log10(θ)
        # 其中 f 是频率 (GHz), θ 是擦地角 (度)
        
        # 简化: 假设擦地角为1度
        grazing_angle = 1.0  # 度
        
        # 频率转换为GHz
        freq_ghz = frequency / 1e9
        
        # 计算σ⁰
        sigma0 = A + B * np.log10(freq_ghz) + C * np.log10(grazing_angle)
        
        # 转换为线性值 (m²/m²)
        return 10**(sigma0 / 10)

class RadarClutterSimulator:
    """雷达杂波模拟器"""
    
    def __init__(self, scene: SceneModel, clutter_model: str = "empirical"):
        """
        :param clutter_model: 杂波模型类型 ('constant', 'empirical')
        """
        self.scene = scene
        
        if clutter_model == "constant":
            self.model = ConstantClutterModel(scene)
        elif clutter_model == "empirical":
            self.model = EmpiricalClutterModel(scene)
        else:
            raise ValueError(f"未知杂波模型: {clutter_model}")
    
    def simulate_clutter_echo(self, radar_position: Tuple[float, float, float],
                             radar_orientation: Tuple[float, float, float],
                             frequency: float, bandwidth: float, prf: float,
                             num_pulses: int) -> np.ndarray:
        """
        模拟杂波回波
        :param radar_position: 雷达位置 (x, y, z) (m)
        :param radar_orientation: 雷达朝向 (方位角, 俯仰角, 滚动角) (度)
        :param frequency: 雷达频率 (Hz)
        :param bandwidth: 雷达带宽 (Hz)
        :param prf: 脉冲重复频率 (Hz)
        :param num_pulses: 脉冲数量
        :return: 杂波回波数据 (num_pulses, num_samples)
        """
        # 简化实现 - 实际应用中应使用更精确的模型
        # 这里返回一个随机噪声作为杂波回波
        num_samples = int(2 * bandwidth / prf)  # 估计采样点数
        return np.random.normal(0, 0.1, (num_pulses, num_samples))
    
    def generate_clutter_map(self, bounds: Tuple[float, float, float, float], 
                            resolution: float, frequency: float) -> np.ndarray:
        """生成杂波RCS地图"""
        return self.model.generate_clutter_map(bounds, resolution, frequency)
    
    def get_clutter_at_position(self, position: Tuple[float, float], frequency: float) -> float:
        """获取指定位置的杂波RCS (m²)"""
        return self.model.get_clutter_rcs(position, frequency)