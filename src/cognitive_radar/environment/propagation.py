import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
from .scene import SceneModel

@dataclass
class PropagationModel:
    """传播模型基类"""
    scene: SceneModel
    
    def calculate_path_loss(self, tx_pos: Tuple[float, float, float],
                           rx_pos: Tuple[float, float, float],
                           frequency: float) -> float:
        """计算路径损耗 (dB)"""
        raise NotImplementedError("子类必须实现此方法")
    
    def calculate_atmospheric_loss(self, distance: float, frequency: float) -> float:
        """计算大气衰减 (dB)"""
        # 基于ITU-R P.676建议书
        # 简化模型: 大气衰减 ≈ α * distance (km)
        
        # 不同频率的衰减系数 (dB/km)
        attenuation_coeffs = {
            1e9: 0.05,    # L波段
            3e9: 0.07,    # S波段
            5e9: 0.12,    # C波段
            10e9: 0.25,   # X波段
            24e9: 0.5,    # K波段
            35e9: 0.8,    # Ka波段
            77e9: 1.5     # W波段
        }
        
        # 找到最接近的频率
        freqs = np.array(list(attenuation_coeffs.keys()))
        idx = np.abs(freqs - frequency).argmin()
        closest_freq = freqs[idx]
        alpha = attenuation_coeffs[closest_freq]
        
        # 考虑天气影响
        precipitation = self.scene.weather.get("precipitation", 0.0)
        if precipitation > 0:
            # 降雨衰减: β * precipitation_rate^γ
            beta, gamma = {
                1e9: (0.0001, 1.0),
                3e9: (0.0002, 1.1),
                5e9: (0.0005, 1.2),
                10e9: (0.001, 1.3),
                24e9: (0.003, 1.4),
                35e9: (0.007, 1.5),
                77e9: (0.015, 1.6)
            }[closest_freq]
            alpha += beta * (precipitation ** gamma)
        
        return alpha * (distance / 1000)  # 转换为km

class FreeSpacePropagation(PropagationModel):
    """自由空间传播模型"""
    
    def calculate_path_loss(self, tx_pos: Tuple[float, float, float],
                           rx_pos: Tuple[float, float, float],
                           frequency: float) -> float:
        """计算自由空间路径损耗 (dB)"""
        # 计算距离
        dx = tx_pos[0] - rx_pos[0]
        dy = tx_pos[1] - rx_pos[1]
        dz = tx_pos[2] - rx_pos[2]
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # 自由空间路径损耗公式: L = 20*log10(4πd/λ)
        wavelength = 3e8 / frequency
        fspl = 20 * np.log10(4 * np.pi * distance / wavelength)
        
        # 添加大气衰减
        atmospheric_loss = self.calculate_atmospheric_loss(distance, frequency)
        
        return fspl + atmospheric_loss

class TerrainAwarePropagation(PropagationModel):
    """地形感知传播模型"""
    
    def calculate_path_loss(self, tx_pos: Tuple[float, float, float],
                           rx_pos: Tuple[float, float, float],
                           frequency: float) -> float:
        """计算地形感知路径损耗 (dB)"""
        # 计算直线距离
        dx = rx_pos[0] - tx_pos[0]
        dy = rx_pos[1] - tx_pos[1]
        dz = rx_pos[2] - tx_pos[2]
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # 自由空间路径损耗
        wavelength = 3e8 / frequency
        fspl = 20 * np.log10(4 * np.pi * distance / wavelength)
        
        # 地形衰减
        terrain_loss = self._calculate_terrain_loss(tx_pos, rx_pos)
        
        # 植被衰减
        vegetation_loss = self._calculate_vegetation_loss(tx_pos, rx_pos, frequency)
        
        # 大气衰减
        atmospheric_loss = self.calculate_atmospheric_loss(distance, frequency)
        
        total_loss = fspl + terrain_loss + vegetation_loss + atmospheric_loss
        return total_loss
    
    def _calculate_terrain_loss(self, tx_pos: Tuple[float, float, float],
                              rx_pos: Tuple[float, float, float]) -> float:
        """计算地形引起的额外损耗 (dB)"""
        # 精确计算路径损耗
        return self._calculate_precise_terrain_loss(tx_pos, rx_pos)
    
    def _calculate_precise_terrain_loss(self, tx_pos, rx_pos) -> float:
        """基于实际路径计算地形损耗"""
        # 路径向量
        vec = np.array(rx_pos) - np.array(tx_pos)
        distance = np.linalg.norm(vec)
        if distance == 0:
            return 0.0
        direction = vec / distance
        
        # 路径采样参数
        num_samples = max(100, int(distance / 10))
        segment_length = distance / num_samples
        
        total_obstruction = 0.0
        
        for i in range(1, num_samples):
            # 计算路径点位置
            current_dist = i * segment_length
            point = np.array(tx_pos) + current_dist * direction
            x, y = point[0], point[1]
            
            # 获取地形高度
            terrain_height = self.scene.get_ground_height(x, y)
            
            # 计算路径高度（沿直线路径）
            # 注意：路径高度是路径上的点的高度，不是地形高度
            path_height = tx_pos[2] + (rx_pos[2] - tx_pos[2]) * (current_dist / distance)
            
            # 检查是否被地形阻挡
            if path_height < terrain_height:
                obstruction_depth = terrain_height - path_height
                total_obstruction += obstruction_depth * (segment_length / distance)
        
        # 将地形遮挡转换为dB损耗 (每米遮挡增加约0.5dB)
        terrain_loss = min(40.0, float(total_obstruction * 0.5 * distance))
        return terrain_loss
    
    def _calculate_vegetation_loss(self, tx_pos: Tuple[float, float, float],
                                 rx_pos: Tuple[float, float, float],
                                 frequency: float) -> float:
        """计算植被引起的额外损耗 (dB)"""
        # 路径向量
        vec = np.array(rx_pos) - np.array(tx_pos)
        distance = np.linalg.norm(vec)
        if distance == 0:
            return 0.0
        direction = vec / distance
        
        # 路径采样参数
        num_samples = max(100, int(distance / 10))
        segment_length = distance / num_samples
        
        total_loss = 0.0
        
        for i in range(num_samples):
            # 计算路径点位置
            current_dist = i * segment_length
            point = np.array(tx_pos) + current_dist * direction
            x, y = point[0], point[1]
            
            # 获取地面高度
            ground_z = self.scene.get_ground_height(x, y)
            
            # 计算路径高度（沿直线路径）
            path_height = tx_pos[2] + (rx_pos[2] - tx_pos[2]) * (current_dist / distance)
            
            # 检查是否在植被区域内且路径低于植被高度
            if path_height < ground_z + 20:  # 假设植被高度不超过20米
                attenuation = self.scene.get_vegetation_attenuation(x, y, frequency)
                total_loss += attenuation * segment_length
        
        return float(total_loss)

class RadarPropagationSimulator:
    """雷达传播模拟器"""
    
    def __init__(self, scene: SceneModel, model_type: str = "terrain"):
        """
        :param model_type: 传播模型类型 ('free_space', 'terrain')
        """
        self.scene = scene
        
        if model_type == "free_space":
            self.model = FreeSpacePropagation(scene)
        elif model_type == "terrain":
            self.model = TerrainAwarePropagation(scene)
        else:
            raise ValueError(f"未知传播模型: {model_type}")
    
    def calculate_path_loss(self, tx_pos: Tuple[float, float, float],
                          rx_pos: Tuple[float, float, float],
                          frequency: float) -> float:
        """计算路径损耗 (dB)"""
        return self.model.calculate_path_loss(tx_pos, rx_pos, frequency)
    
    def calculate_signal_strength(self, tx_power: float, tx_gain: float, rx_gain: float,
                                 tx_pos: Tuple[float, float, float],
                                 rx_pos: Tuple[float, float, float],
                                 frequency: float) -> float:
        """
        计算接收信号强度 (dBm)
        :param tx_power: 发射功率 (W)
        :param tx_gain: 发射天线增益 (dBi)
        :param rx_gain: 接收天线增益 (dBi)
        :param tx_pos: 发射位置 (x, y, z) (m)
        :param rx_pos: 接收位置 (x, y, z) (m)
        :param frequency: 频率 (Hz)
        :return: 接收信号强度 (dBm)
        """
        # 计算路径损耗
        path_loss = self.calculate_path_loss(tx_pos, rx_pos, frequency)
        
        # 计算接收功率
        rx_power_dbm = 10 * np.log10(tx_power * 1000) + tx_gain + rx_gain - path_loss
        
        return rx_power_dbm
    
    def calculate_max_range(self, tx_power: float, tx_gain: float, rx_gain: float,
                          frequency: float, sensitivity: float) -> float:
        """
        计算最大探测距离 (m)
        :param sensitivity: 接收机灵敏度 (dBm)
        """
        # 使用迭代法求解
        min_range = 100  # 最小距离 (m)
        max_range = 100000  # 最大距离 (m)
        tx_pos = (0, 0, 10)  # 假设雷达位置
        rx_pos = (0, 0, 10)  # 假设单站雷达
        
        # 二分查找
        for _ in range(20):  # 最多迭代20次
            mid_range = (min_range + max_range) / 2
            rx_pos = (mid_range, 0, 10)  # 假设目标在同一高度
            
            # 计算接收信号强度
            rx_power = self.calculate_signal_strength(
                tx_power, tx_gain, rx_gain, tx_pos, rx_pos, frequency
            )
            
            if rx_power > sensitivity:
                min_range = mid_range
            else:
                max_range = mid_range
        
        return (min_range + max_range) / 2