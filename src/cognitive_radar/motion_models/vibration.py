import numpy as np
from .base import KinematicModel

class VibrationMotion(KinematicModel):
    """振动运动模型"""
    
    def __init__(self, base_position, amplitudes, frequencies, phases=(0, 0, 0)):
        """
        :param base_position: 基础位置 (x, y, z)
        :param amplitudes: 三轴振幅 (米)
        :param frequencies: 三轴频率 (Hz)
        :param phases: 三轴相位 (弧度)
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