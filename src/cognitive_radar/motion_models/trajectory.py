from .base import KinematicModel
import numpy as np

class TrajectoryMotion(KinematicModel):
    """基于轨迹文件的运动模型"""
    
    def __init__(self, trajectory_file, time_col='timestamp', 
                 pos_cols=('x', 'y', 'z'), rot_cols=('yaw', 'pitch', 'roll'),
                 interpolate='linear'):
        """
        :param trajectory_file: 轨迹数据文件路径
        :param time_col: 时间戳列名
        :param pos_cols: 位置列名元组
        :param rot_cols: 旋转列名元组
        :param interpolate: 插值方式
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
        self.pos_interpolator = interp1d(
            self.times, self.positions, axis=0,
            kind=interpolate, bounds_error=False,
            fill_value=0.0
        )

        if rot_cols:
            self.rot_interpolator = interp1d(
                self.times, self.rotations, axis=0,
                kind=interpolate, bounds_error=False,
                fill_value=0.0
            )
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