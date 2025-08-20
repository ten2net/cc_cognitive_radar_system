from .base import KinematicModel
from typing import List

class CompositeMotion(KinematicModel):
    """复合运动模型（多运动模型叠加）"""
    
    def __init__(self, motion_models: List[KinematicModel]):
        """
        :param motion_models: 运动模型列表，按优先级从高到低排序
        """
        self.motion_models = motion_models

    def get_state(self, t):
        # 第一个模型为基准
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
    
    def __init__(self, leader_motion: KinematicModel, relative_func):
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