import numpy as np
from .base import KinematicModel

class LinearMotion(KinematicModel):
    """匀速直线运动模型"""
    
    def __init__(self, start_position, velocity, rotation=(0, 0, 0)):
        """
        :param start_position: 初始位置 (x, y, z)
        :param velocity: 速度向量
        :param rotation: 初始旋转角度
        """
        self.position = np.array(start_position)
        self.velocity = np.array(velocity)
        self.rotation = np.array(rotation)
        self.start_time = 0

    def get_state(self, t):
        dt = t - self.start_time
        return {
            'location': self.position + self.velocity * dt,
            'rotation': self.rotation.copy(),
            'speed': self.velocity.copy()
        }

    def update(self, dt):
        self.position += self.velocity * dt
        self.start_time += dt


class AcceleratedMotion(LinearMotion):
    """匀加速直线运动模型"""
    
    def __init__(self, start_position, start_velocity, acceleration, rotation=(0, 0, 0)):
        """
        :param start_position: 初始位置
        :param start_velocity: 初始速度
        :param acceleration: 加速度向量
        :param rotation: 初始旋转角度
        """
        super().__init__(start_position, start_velocity, rotation)
        self.acceleration = np.array(acceleration)

    def get_state(self, t):
        dt = t - self.start_time
        return {
            'location': self.position + self.velocity * dt + 0.5 * self.acceleration * dt**2,
            'rotation': self.rotation.copy(),
            'speed': self.velocity + self.acceleration * dt
        }

    def update(self, dt):
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt**2
        self.velocity += self.acceleration * dt
        self.start_time += dt