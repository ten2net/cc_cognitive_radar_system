import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class Terrain:
    """地形数据类"""
    elevation: np.ndarray  # 高程数据 (m)
    resolution: float      # 分辨率 (m/像素)
    bounds: Tuple[float, float, float, float]  # 边界 (min_x, min_y, max_x, max_y)
    terrain_type: str = "unknown"  # 地形类型 ('flat', 'mountain', 'urban', etc.)
    
    def get_elevation_at(self, x: float, y: float) -> float:
        """获取指定位置的高程"""
        # 计算网格索引
        col = int((x - self.bounds[0]) / self.resolution)
        row = int((y - self.bounds[1]) / self.resolution)
        
        # 边界检查
        if 0 <= row < self.elevation.shape[0] and 0 <= col < self.elevation.shape[1]:
            return self.elevation[row, col]
        return 0.0  # 默认高程

@dataclass
class Building:
    """建筑物模型"""
    id: str
    position: Tuple[float, float]  # (x, y) 中心位置
    height: float
    dimensions: Tuple[float, float, float]  # (长, 宽, 高) (m)
    material: str = "concrete"  # 建筑材料
    rcs: float = 100.0  # 平均RCS (m²)
    
    def contains_point(self, x: float, y: float) -> bool:
        """检查点是否在建筑物内"""
        half_length, half_width, _ = [d/2 for d in self.dimensions]
        min_x = self.position[0] - half_length
        max_x = self.position[0] + half_length
        min_y = self.position[1] - half_width
        max_y = self.position[1] + half_width
        return min_x <= x <= max_x and min_y <= y <= max_y

@dataclass
class Vegetation:
    """植被模型"""
    
    def __init__(self, id: str, position: Tuple[float, float], area: float, 
                 height: float, density: float, type: str = "forest"):
        """
        :param id: 植被ID
        :param position: 中心位置 (x, y)
        :param area: 面积 (m²)
        :param height: 高度 (m)
        :param density: 密度 (0-1)
        :param type: 植被类型 ('urban', 'forest', 'woodland', 'grass', 'crops', 'shrubs')
        """
        self.id = id
        self.position = position
        self.area = area
        self.height = height
        self.density = density
        self.type = type
    
    def get_attenuation(self, frequency: float) -> float:
        """计算植被衰减 (dB/m)"""
        # 不同植被类型的衰减模型参数
        attenuation_models = {
            "urban": 0.22,  # 增加城市基础衰减系数
            "forest": 0.20,
            "woodland": 0.15,
            "grass": 0.05,
            "crops": 0.10,
            "shrubs": 0.12
        }
        
        # 频率依赖因子 (f in GHz)
        freq_ghz = frequency / 1e9
        frequency_factors = {
            "urban": freq_ghz**0.6,
            "forest": freq_ghz**0.5,
            "woodland": freq_ghz**0.45,
            "grass": freq_ghz**0.3,
            "crops": freq_ghz**0.4,
            "shrubs": freq_ghz**0.5
        }
        
        # 获取基础衰减系数
        base_attenuation = attenuation_models.get(self.type, 0.1)
        
        # 获取频率因子
        freq_factor = frequency_factors.get(self.type, 1.0)
        
        # 密度因子 (城市植被密度影响更大)
        if self.type == "urban":
            density_factor = 0.7 + 0.3 * self.density  # 0.7-1.0
        else:
            density_factor = 0.5 + 0.5 * self.density  # 0.5-1.0
        
        # 计算总衰减
        return base_attenuation * freq_factor * density_factor
    
    def contains_point(self, x: float, y: float) -> bool:
        """
        检查点是否在植被区域内
        :param x: X坐标
        :param y: Y坐标
        :return: 是否在区域内
        """
        # 简化的圆形区域检查
        distance = np.sqrt((x - self.position[0])**2 + (y - self.position[1])**2)
        radius = np.sqrt(self.area / np.pi)  # 等效半径
        return distance <= radius

@dataclass
class SceneModel:
    """场景模型
        地形建模​​：支持高程数据加载和查询
        ​​建筑物建模​​：包含位置、尺寸和材料属性
        ​​植被建模​​：支持不同类型植被的衰减计算
        目标管理：管理场景中的目标
        ​​天气参数​​：温度、湿度、气压和降水量    
    """
    terrain: Terrain
    buildings: Dict[str, Building] = field(default_factory=dict)
    vegetation: Dict[str, Vegetation] = field(default_factory=dict)
    targets: Dict[str, Dict] = field(default_factory=dict)  # 新增：目标字典
    weather: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 20.0,  # 温度 (°C)
        "humidity": 50.0,     # 湿度 (%)
        "pressure": 1013.25,  # 气压 (hPa)
        "precipitation": 0.0   # 降水量 (mm/h)
    })
    
    def add_building(self, building: Building):
        """添加建筑物"""
        self.buildings[building.id] = building
    
    def add_vegetation(self, vegetation: Vegetation):
        """添加植被"""
        self.vegetation[vegetation.id] = vegetation
        
    # 添加目标管理方法
    def add_target(self, target: Dict):
        """添加目标"""
        if 'id' not in target:
            target['id'] = f"target_{len(self.targets)+1}"
        self.targets[target['id']] = target        
    
    def update_weather(self, **kwargs):
        """更新天气参数"""
        self.weather.update(kwargs)
    
    def get_ground_height(self, x: float, y: float) -> float:
        """获取指定位置的地面高度（包括建筑物）"""
        # 转换为网格坐标
        grid_x = int((x - self.terrain.bounds[0]) / self.terrain.resolution)
        grid_y = int((y - self.terrain.bounds[1]) / self.terrain.resolution)
        
        # 确保在网格范围内
        if 0 <= grid_x < self.terrain.elevation.shape[0] and 0 <= grid_y < self.terrain.elevation.shape[1]:
            height = self.terrain.elevation[grid_x, grid_y]
            
            # 检查是否有建筑物
            for building in self.buildings.values():
                if building.contains_point(x, y):
                    height += building.height
                    
            return height
        else:
            return 0.0  # 边界外默认为0高度
    
    def get_vegetation_attenuation(self, x: float, y: float, frequency: float) -> float:
        """获取植被衰减 (dB/m)"""
        for veg in self.vegetation.values():
            # 简化的包含检查 (实际应用中应使用更精确的方法)
            if np.sqrt((x - veg.position[0])**2 + (y - veg.position[1])**2) < np.sqrt(veg.area/np.pi):
                return veg.get_attenuation(frequency)
        return 0.0
    
    def get_building_at_position(self, x: float, y: float) -> Optional[Building]:
        """获取指定位置的建筑物"""
        for building in self.buildings.values():
            if building.contains_point(x, y):
                return building
        return None
    
    def to_dict(self) -> Dict:
        """转换为字典表示（修复：添加地形高程数据）"""
        return {
            "terrain": {
                "elevation": self.terrain.elevation.tolist(),  # 新增：保存高程数据
                "resolution": self.terrain.resolution,
                "bounds": self.terrain.bounds,
                "terrain_type": self.terrain.terrain_type
            },
            "buildings": {id: vars(b) for id, b in self.buildings.items()},
            "vegetation": {id: vars(v) for id, v in self.vegetation.items()},
            "targets": self.targets,  # 新增：添加目标
            "weather": self.weather
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """从字典创建场景模型（修复：恢复地形高程数据）"""
        terrain_data = data["terrain"]
        terrain = Terrain(
            elevation=np.array(terrain_data["elevation"]),  # 修复：加载实际高程
            resolution=terrain_data["resolution"],
            bounds=tuple(terrain_data["bounds"]),
            terrain_type=terrain_data["terrain_type"]
        )
        
        scene = cls(terrain=terrain, weather=data["weather"])
        
        for id, b_data in data["buildings"].items():
            scene.add_building(Building(
                id=id,
                position=tuple(b_data["position"]),
                height=b_data["height"],
                dimensions=tuple(b_data["dimensions"]),
                material=b_data["material"],
                rcs=b_data["rcs"]
            ))
        
        for id, v_data in data["vegetation"].items():
            scene.add_vegetation(Vegetation(
                id=id,
                position=tuple(v_data["position"]),
                area=v_data["area"],
                height=v_data["height"],
                density=v_data["density"],
                type=v_data["type"]
            ))
        scene.targets = data.get("targets", {})  # 新增：恢复目标
        
        return scene