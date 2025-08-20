import numpy as np
from .base import KinematicModel

class StaticModel(KinematicModel):
    """静态目标模型"""
    
    def __init__(self, position, rotation=None, speed=None):
        """
        :param position: 目标位置 (x, y, z)
        :param rotation: 旋转角度 (yaw, pitch, roll)
        :param speed: 速度向量
        """
        self.position = np.array(position)
        self.rotation = np.array(rotation) if rotation is not None else np.zeros(3)
        self.speed = np.array(speed) if speed is not None else np.zeros(3)

    def get_state(self, t):
        return {
            'location': self.position.copy(),
            'rotation': self.rotation.copy(),
            'speed': self.speed.copy()
        }