from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

class KinematicModel(ABC):
    """运动模型基类"""
    
    @abstractmethod
    def get_state(self, t) -> Dict:
        """获取目标在时间t的状态
        :param t: 时间戳
        :return: 状态字典，包含位置、旋转、速度等信息
        """
        pass

    def update(self, dt):
        """更新模型时间步
        :param dt: 时间增量
        """
        pass