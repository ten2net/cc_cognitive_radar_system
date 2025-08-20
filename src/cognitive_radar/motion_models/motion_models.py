from typing import Dict
import numpy as np
from abc import ABC, abstractmethod


class KinematicModel(ABC):
    """运动模型基类"""

    @abstractmethod
    def get_state(self, t) -> Dict:
        """获取目标在时间t的状态"""
        pass

    def update(self, dt):
        """更新模型时间步"""
        pass


class StaticModel(KinematicModel):
    """静态目标模型"""

    def __init__(self, position, rotation=None, speed=None):
        self.position = np.array(position)
        self.rotation = np.array(
            rotation) if rotation is not None else np.zeros(3)
        self.speed = np.array(speed) if speed is not None else np.zeros(3)

    def get_state(self, t):
        return {
            'location': self.position.copy(),
            'rotation': self.rotation.copy(),
            'speed': self.speed.copy()
        }


class LinearMotion(KinematicModel):
    """匀速直线运动模型"""

    def __init__(self, start_position, velocity, rotation=(0, 0, 0)):
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


class AcceleratedMotion(KinematicModel):
    """匀加速直线运动模型"""

    def __init__(self, start_position, start_velocity, acceleration, rotation=(0, 0, 0)):
        self.position = np.array(start_position)
        self.velocity = np.array(start_velocity)
        self.acceleration = np.array(acceleration)
        self.rotation = np.array(rotation)
        self.start_time = 0

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

        # 计算朝向 - 面向运动方向
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


class VibrationMotion(KinematicModel):
    """振动运动模型"""

    def __init__(self, base_position, amplitudes, frequencies, phases=(0, 0, 0)):
        """
        :param base_position: 基础位置 (x, y, z)
        :param amplitudes: 三轴振幅 (米) (x_amp, y_amp, z_amp)
        :param frequencies: 三轴频率 (Hz) (x_freq, y_freq, z_freq)
        :param phases: 三轴相位 (弧度) (x_phase, y_phase, z_phase)
        """
        self.base_position = np.array(base_position)
        self.amplitudes = np.array(amplitudes)
        self.frequencies = np.array(frequencies)
        self.phases = np.array(phases)
        self.start_time = 0

    def get_state(self, t):
        dt = t - self.start_time
        offset = self.amplitudes * \
            np.sin(2 * np.pi * self.frequencies * dt + self.phases)
        return {
            'location': self.base_position + offset,
            'rotation': np.zeros(3),
            'speed': 2 * np.pi * self.frequencies * self.amplitudes *
            np.cos(2 * np.pi * self.frequencies * dt + self.phases)
        }

    def update(self, dt):
        self.start_time += dt


class CompositeMotion(KinematicModel):
    """复合运动模型（多运动模型叠加）"""

    def __init__(self, motion_models):
        """
        :param motion_models: 运动模型列表，按优先级从高到低排序
        """
        self.motion_models = motion_models

    def get_state(self, t):
        # 第一个模型为基准，其他模型作为偏移量叠加
        base_state = self.motion_models[0].get_state(t)

        # 叠加其他运动模型
        for motion in self.motion_models[1:]:
            offset = motion.get_state(t)
            for key in ['location', 'rotation', 'speed']:
                if key in base_state and key in offset:
                    base_state[key] += offset[key]

        return base_state

    def update(self, dt):
        for motion in self.motion_models:
            motion.update(dt)


class PhaseLockedMotion(KinematicModel):
    """相位锁定运动模型（多目标协同运动）"""

    def __init__(self, leader_motion, relative_func):
        """
        :param leader_motion: 领航目标的运动模型
        :param relative_func: 计算相对位置和朝向的函数
               func(leader_state) -> dict: {'location', 'rotation', 'speed'}
        """
        self.leader_motion = leader_motion
        self.relative_func = relative_func

    def get_state(self, t):
        leader_state = self.leader_motion.get_state(t)
        relative = self.relative_func(leader_state)

        # 合成最终状态
        state = {}
        for key in ['location', 'rotation', 'speed']:
            if key in leader_state and key in relative:
                state[key] = leader_state[key] + relative[key]

        return state

    def update(self, dt):
        self.leader_motion.update(dt)


class TrajectoryMotion(KinematicModel):
    """基于轨迹文件的运动模型"""

    def __init__(self, trajectory_file, time_col='timestamp', pos_cols=('x', 'y', 'z'),
                 rot_cols=('yaw', 'pitch', 'roll'), interpolate='linear'):
        """
        :param trajectory_file: 轨迹数据文件路径（CSV或其他格式）
        :param time_col: 时间戳列名
        :param pos_cols: 位置列名元组
        :param rot_cols: 旋转列名元组
        :param interpolate: 插值方式 ('linear', 'cubic', 'nearest')
        """
        import pandas as pd
        from scipy.interpolate import interp1d

        # 加载轨迹数据
        self.traj = pd.read_csv(trajectory_file)
        self.times = self.traj[time_col].values
        self.positions = self.traj[list(pos_cols)].values

        if rot_cols:
            self.rotations = self.traj[list(rot_cols)].values
        else:
            self.rotations = np.zeros_like(self.positions)

        # 创建插值函数
        self.pos_interpolator = interp1d(self.times, self.positions, axis=0,
                                         kind=interpolate, bounds_error=False,
                                         fill_value=0.0)

        if rot_cols:
            self.rot_interpolator = interp1d(self.times, self.rotations, axis=0,
                                             kind=interpolate, bounds_error=False,
                                             fill_value=0.0)
        else:
            self.rot_interpolator = None

    def get_state(self, t):
        state = {'location': self.pos_interpolator(t).flatten()}

        if self.rot_interpolator:
            state['rotation'] = self.rot_interpolator(t).flatten()

        # 速度通过位置微分计算
        delta_t = 0.001  # 时间差 (1ms)
        pos1 = self.pos_interpolator(t)
        pos2 = self.pos_interpolator(t + delta_t)
        state['speed'] = (pos2 - pos1) / delta_t

        return state

    def update(self, dt):
        pass
