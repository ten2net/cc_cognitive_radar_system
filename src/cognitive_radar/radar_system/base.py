from typing import Dict, List, Tuple
from radarsimpy import Radar
from abc import ABC, abstractmethod

class RadarFactory(ABC):
    """雷达工厂基类"""

    @abstractmethod
    def list_available_radars(self) -> List[str]:
        """获取工厂能生产的雷达类型列表"""
        pass
    @abstractmethod
    def get_radar_info(self, radar_type:str) -> Dict:
        """查看指定类型雷达的参数"""
        pass
    @abstractmethod
    def create(self, 
                     radar_type, 
                     location: Tuple[float, float, float] = (0,0,0),  
                     speed: Tuple[float, float, float] = (0,0,0)) -> Radar :
      pass  