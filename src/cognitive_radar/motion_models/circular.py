import numpy as np
from .base import KinematicModel

class CircularMotion(KinematicModel):
    """圆周运动模型"""
    
    def __init__(self, center, radius, angular_vel, axis='z', start_angle=0, pitch=0):
        """
        :param center: 圆心位置 (x, y, z)
        :param radius: 运动半径 (米)
        :param angular_vel: 角速度 (弧度/秒)
        :param axis: 旋转轴 ('x', 'y', 'z')
        :param start_angle: 初始角度 (弧度)
        :param pitch: 爬升角 (弧度)
        """
        self.center = np.array(center)
        self.radius = radius
        self.angular_vel = angular_vel
        self.axis = axis.lower()
        self.start_angle = start_angle
        self.pitch = pitch
        self.start_time = 0

        # 计算旋转平面
        if self.axis == 'x':
            self.rotation_plane = [1, 2]  # y-z平面
        elif self.axis == 'y':
            self.rotation_plane = [0, 2]  # x-z平面
        else:  # z轴
            self.rotation_plane = [0, 1]  # x-y平面

    def get_state(self, t):
        dt = t - self.start_time
        angle = self.start_angle + self.angular_vel * dt
        rot_plane = self.rotation_plane

        # 计算圆周位置
        location = self.center.copy()
        location[rot_plane[0]] += self.radius * np.cos(angle)
        location[rot_plane[1]] += self.radius * np.sin(angle)
        location[2] += np.sin(self.pitch) * self.radius * dt * self.angular_vel

        # 计算朝向
        rotation = np.zeros(3)
        rotation[0] = self.pitch * 180 / np.pi  # 俯仰角

        # 计算速度
        speed = np.zeros(3)
        speed[rot_plane[0]] = -self.radius * self.angular_vel * np.sin(angle)
        speed[rot_plane[1]] = self.radius * self.angular_vel * np.cos(angle)
        speed[2] = np.sin(self.pitch) * self.radius * self.angular_vel

        return {
            'location': location,
            'rotation': rotation,
            'speed': speed
        }

    def update(self, dt):
        self.start_time += dt