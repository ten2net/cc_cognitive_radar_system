import numpy as np
from enum import Enum
from scipy.interpolate import CubicSpline, interp1d
import pandas as pd

class MotionModelType(Enum):
    """支持的动态目标运动模型类型"""
    SINUSOIDAL = "sinusoidal"
    BIRD = "bird"
    LOW_SPEED_DRONE = "low_speed_drone"
    HIGH_SPEED_DRONE = "high_speed_drone"
    PARAMETRIC = "parametric"
    PHYSICS_BASED = "physics_based"
    RANDOM_WALK = "random_walk"
    BEHAVIOR_BASED = "behavior_based"
    MOTION_CAPTURE = "motion_capture"
    SWARM = "swarm"

class MotionModelFactory:
    """动态目标模型工厂类，用于创建各种类型的运动模型"""
    
    @staticmethod
    def create(model_type, **kwargs):
        """
        创建指定类型的运动模型
        
        Args:
            model_type: MotionModelType 枚举值，指定要创建的运动模型类型
            **kwargs: 模型特定的参数
        
        Returns:
            目标位置函数
        """
        model_creators = {
            MotionModelType.SINUSOIDAL: MotionModelFactory._create_sinusoidal,
            MotionModelType.BIRD: MotionModelFactory._create_bird,
            MotionModelType.LOW_SPEED_DRONE: MotionModelFactory._create_low_speed_drone,
            MotionModelType.HIGH_SPEED_DRONE: MotionModelFactory._create_high_speed_drone,
            MotionModelType.PARAMETRIC: MotionModelFactory._create_parametric,
            MotionModelType.PHYSICS_BASED: MotionModelFactory._create_physics_based,
            MotionModelType.RANDOM_WALK: MotionModelFactory._create_random_walk,
            MotionModelType.BEHAVIOR_BASED: MotionModelFactory._create_behavior_based,
            MotionModelType.MOTION_CAPTURE: MotionModelFactory._create_motion_capture,
            MotionModelType.SWARM: MotionModelFactory._create_swarm
        }
        
        if model_type not in model_creators:
            raise ValueError(f"Unknown motion model type: {model_type}")
            
        return model_creators[model_type](**kwargs)
    
    @staticmethod
    def _create_sinusoidal(
        base_position=(100, 0, 0),
        frequency=2.0,
        amplitude=5.0,
        axis="z"
    ):
        """
        创建正确的正弦振荡运动模型
        
        Args:
            base_position: 基础位置 (x, y, z)
            frequency: 振荡频率 (Hz)
            amplitude: 振荡幅度 (米)
            axis: 振荡轴 ('x', 'y', 'z')
        """
        # 将基础位置转为NumPy数组以便操作
        base_arr = np.array(base_position)
        
        def position(t):
            # 关键修复：每次创建新数组副本
            pos = base_arr.copy()  # 避免修改原始数组
            phase = 2 * np.pi * frequency * t
            oscillation = amplitude * np.sin(phase)
            
            # 更清晰的轴选择逻辑
            axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis.lower(), 2)
            pos[axis_idx] += oscillation
            
            return pos
        
        return position
    
    @staticmethod
    def _create_bird(
        base_position=(100, 0, 50),
        wing_flap_freq=5.0,
        wing_flap_amp=0.5,
        flight_speed=15,
        direction_change_freq=0.2,
        turbulence_amp=1.0
    ):
        """
        创建飞鸟运动模型
        
        Args:
            base_position: 基础位置 (x, y, z)
            wing_flap_freq: 翅膀扑动频率 (Hz)
            wing_flap_amp: 翅膀扑动幅度 (米)
            flight_speed: 飞行速度 (m/s)
            direction_change_freq: 方向变化频率 (Hz)
            turbulence_amp: 湍流扰动幅度 (米)
        """
        x0, y0, z0 = base_position
        
        def position(t):
            # 翅膀扑动效果
            z_flap = wing_flap_amp * np.sin(2 * np.pi * wing_flap_freq * t)
            
            # 水平飞行路径
            theta = 0.1 * np.sin(2 * np.pi * direction_change_freq * t)
            dx = flight_speed * np.cos(theta) * t
            dy = flight_speed * np.sin(theta) * t
            
            # 湍流扰动
            turb_x = turbulence_amp * (0.5 - np.random.rand())
            turb_y = turbulence_amp * (0.5 - np.random.rand())
            turb_z = turbulence_amp * (0.5 - np.random.rand())
            
            return np.array([
                x0 + dx + turb_x,
                y0 + dy + turb_y,
                z0 + z_flap + turb_z
            ])
        
        return position
    
    @staticmethod
    def _create_low_speed_drone(
        center_position=(150, 0, 40),
        orbit_radius=20,
        orbit_freq=0.1,
        hover_amp=2.0,
        hover_freq=0.5,
        speed=8
    ):
        """
        创建低速无人机运动模型
        
        Args:
            center_position: 中心位置 (x, y, z)
            orbit_radius: 盘旋半径 (米)
            orbit_freq: 盘旋频率 (Hz)
            hover_amp: 悬停振荡幅度 (米)
            hover_freq: 悬停振荡频率 (Hz)
            speed: 平均飞行速度 (m/s)
        """
        x0, y0, z0 = center_position
        
        def position(t):
            # 盘旋运动
            orbit_angle = 2 * np.pi * orbit_freq * t
            x = x0 + orbit_radius * np.cos(orbit_angle)
            y = y0 + orbit_radius * np.sin(orbit_angle)
            
            # 悬停振荡
            z = z0 + hover_amp * np.sin(2 * np.pi * hover_freq * t)
            
            # 缓慢线性漂移
            drift = speed * t * 0.2
            return np.array([
                x + drift,
                y,
                z
            ])
        
        return position
    
    @staticmethod
    def _create_high_speed_drone(
        start_position=(0, 50, 100),
        end_position=(500, 300, 150),
        cruise_speed=60,
        maneuver_amp=20,
        maneuver_freq=0.5,
        jitter_amp=0.3
    ):
        """
        创建高速无人机运动模型
        
        Args:
            start_position: 起始位置 (x, y, z)
            end_position: 目标位置 (x, y, z)
            cruise_speed: 巡航速度 (m/s)
            maneuver_amp: 机动幅度 (米)
            maneuver_freq: 机动频率 (Hz)
            jitter_amp: 机体抖动幅度 (米)
        """
        x1, y1, z1 = start_position
        x2, y2, z2 = end_position
        
        # 计算总距离和飞行方向
        distance = np.linalg.norm(np.array(end_position) - np.array(start_position))
        flight_duration = distance / cruise_speed
        
        def position(t):
            # 直线飞行路径
            flight_progress = min(t / flight_duration, 1.0)
            base_pos = np.array(start_position) + flight_progress * (np.array(end_position) - np.array(start_position))
            
            # 高G机动
            maneuver_x = maneuver_amp * np.sin(2 * np.pi * maneuver_freq * t)
            maneuver_y = maneuver_amp * np.sin(2 * np.pi * maneuver_freq * t + np.pi/2)
            
            # 机体抖动
            jitter = jitter_amp * (np.random.rand(3) - 0.5)
            
            return np.array([
                base_pos[0] + maneuver_x + jitter[0],
                base_pos[1] + maneuver_y + jitter[1],
                base_pos[2] + jitter[2]
            ])
        
        return position
    
    @staticmethod
    def _create_parametric(
        waypoints,
        times,
        motion_type="smooth"
    ):
        """
        创建参数化路径运动模型
        
        Args:
            waypoints: 路径点列表 [(x1,y1,z1), (x2,y2,z2), ...]
            times: 到达各路径点的时间 [t1, t2, ...]
            motion_type: 'smooth'(样条) | 'linear' | 'accel'(加速度)
        """
        waypoints = np.array(waypoints)
        times = np.array(times)
        
        if motion_type == "smooth":
            spl_x = CubicSpline(times, waypoints[:, 0])
            spl_y = CubicSpline(times, waypoints[:, 1])
            spl_z = CubicSpline(times, waypoints[:, 2])
        
        def position(t):
            if motion_type == "linear":
                idx = np.searchsorted(times, t) - 1
                if idx < 0:
                    return waypoints[0]
                elif idx >= len(times) - 1:
                    return waypoints[-1]
                    
                t0, t1 = times[idx], times[idx+1]
                ratio = (t - t0) / (t1 - t0)
                return waypoints[idx] + ratio * (waypoints[idx+1] - waypoints[idx])
            
            elif motion_type == "accel":
                idx = np.searchsorted(times, t) - 1
                if idx < 0:
                    return waypoints[0]
                if idx >= len(times) - 1:
                    return waypoints[-1]
                
                t0, t1 = times[idx], times[idx+1]
                tau = (t - t0) / (t1 - t0)
                # 缓入缓出函数
                alpha = 3*tau**2 - 2*tau**3
                return waypoints[idx] + alpha * (waypoints[idx+1] - waypoints[idx])
            
            else:
                return np.array([spl_x(t), spl_y(t), spl_z(t)]) # type: ignore
        
        return position
    
    @staticmethod
    def _create_physics_based(
        initial_state=np.array([0, 0, 0, 0, 0, 0]),
        mass=1.0,
        max_thrust=15.0,
        drag_coeff=0.1
    ):
        """
        创建基于物理的动力学模型
        
        Args:
            initial_state: 初始状态 [x, y, z, vx, vy, vz]
            mass: 质量 (kg)
            max_thrust: 最大推力 (N)
            drag_coeff: 空气阻力系数
        """
        state = initial_state.copy()
        last_t = 0
        
        def position(t):
            nonlocal state, last_t
            dt = t - last_t
            if dt <= 0:
                return state[:3].copy()
                
            # 状态: [x, y, z, vx, vy, vz]
            pos = state[:3]
            vel = state[3:]
            
            # 简单控制器 - 移动到目标位置
            target = np.array([200, 0, 100])  # 目标位置
            error = target - pos
            control = 0.5 * error
            
            # 计算推力
            thrust = np.array([
                control[0] * max_thrust,
                control[1] * max_thrust,
                control[2] * max_thrust + 9.8 * mass  # 补偿重力
            ])
            
            # 阻力 (速度方向相反)
            drag = -drag_coeff * vel * np.linalg.norm(vel)
            
            # 加速度 = (推力 + 阻力) / 质量
            accel = (thrust + drag) / mass
            
            # 数值积分 (欧拉方法)
            new_vel = vel + accel * dt
            new_pos = pos + vel * dt
            
            # 更新状态
            state = np.concatenate([new_pos, new_vel])
            last_t = t
            return new_pos.copy()
        
        return position
    
    @staticmethod
    def _create_random_walk(
        initial_pos=(0, 0, 50),
        speed_mean=10.0,
        speed_std=3.0,
        turn_rate=0.2,
        rng_seed=None
    ):
        """
        创建随机游走运动模型
        
        Args:
            initial_pos: 初始位置 (x, y, z)
            speed_mean: 平均速度 (m/s)
            speed_std: 速度标准差
            turn_rate: 转弯概率 (每秒)
            rng_seed: 随机数种子
        """
        rng = np.random.default_rng(rng_seed)
        last_pos = np.array(initial_pos)
        last_t = 0
        current_speed = rng.normal(speed_mean, speed_std)
        current_dir = rng.uniform(0, 2*np.pi)  # 水平方向
        current_pitch = rng.uniform(-0.1, 0.1)  # 俯仰角
        
        def position(t):
            nonlocal last_pos, last_t, current_speed, current_dir, current_pitch
            
            dt = t - last_t
            if dt <= 0:
                return last_pos
            
            # 随机改变方向
            if rng.random() < turn_rate * dt:
                current_dir = rng.uniform(0, 2*np.pi)
                current_pitch = rng.uniform(-0.2, 0.2)
                current_speed = rng.normal(speed_mean, speed_std)
            
            # 计算位移
            dxy = current_speed * dt * np.array([np.cos(current_dir), np.sin(current_dir)])
            dz = current_speed * dt * np.sin(current_pitch)
            
            # 更新位置
            new_pos = last_pos + np.array([dxy[0], dxy[1], dz])
            last_pos = new_pos
            last_t = t
            return new_pos
        
        return position
    
    @staticmethod
    def _create_behavior_based(
        initial_pos=(0, 0, 100)
    ):
        """
        创建基于状态机的行为模型
        
        Args:
            initial_pos: 初始位置 (x, y, z)
        """
        from enum import Enum, auto
        import numpy as np
        
        class BehaviorState(Enum):
            CRUISE = auto()
            EVADE = auto()
            LOITER = auto()
            LANDING = auto()
        
        state = BehaviorState.CRUISE
        last_t = 0
        pos = np.array(initial_pos)
        speed = 20.0
        target = np.array([200, 0, 50])
        threat_pos = np.array([150, 30, 0])  # 假想的雷达威胁位置
        
        def position(t):
            nonlocal state, last_t, pos, target, speed
            dt = t - last_t
            if dt <= 0:
                return pos
            
            # 状态转换逻辑
            dist_to_threat = np.linalg.norm(pos[:2] - threat_pos[:2])
            if state == BehaviorState.CRUISE and dist_to_threat < 100:
                state = BehaviorState.EVADE
                # 计算逃逸方向 (远离威胁)
                escape_dir = (pos - threat_pos)[:2]
                escape_dir /= np.linalg.norm(escape_dir)
                target[:2] = pos[:2] + escape_dir * 200
                target[2] = pos[2] + 30  # 同时爬升
                speed = 30.0
                
            elif state == BehaviorState.EVADE and dist_to_threat > 150:
                state = BehaviorState.CRUISE
                target = np.array([300, 50, 60])
                speed = 25.0
            
            # 状态行为实现
            if state == BehaviorState.CRUISE:
                # 简单朝向目标移动
                dir_vec = (target - pos)
                step = dir_vec / np.linalg.norm(dir_vec) * min(speed * dt, np.linalg.norm(dir_vec)) # type: ignore
                pos += step
            
            elif state == BehaviorState.EVADE:
                # 以锯齿方式远离威胁
                dir_vec = (target - pos)
                step = dir_vec / np.linalg.norm(dir_vec) * min(speed * dt, np.linalg.norm(dir_vec)) # type: ignore
                # 添加随机扰动
                pos += step + np.random.normal(0, 2, 3)
            
            last_t = t
            return pos.copy()
        
        return position
    
    @staticmethod
    def _create_motion_capture(
        csv_file,
        time_scale=1.0,
        time_column="time",
        x_column="x",
        y_column="y",
        z_column="z"
    ):
        """
        创建基于运动捕捉数据的模型
        
        Args:
            csv_file: 包含轨迹数据的CSV文件
            time_scale: 时间缩放因子
            time_column: 时间列名
            x_column: X坐标列名
            y_column: Y坐标列名
            z_column: Z坐标列名
        """
        df = pd.read_csv(csv_file)
        times = df[time_column].values * time_scale # type: ignore
        x = df[x_column].values
        y = df[y_column].values
        z = df[z_column].values
        
        # 创建插值函数
        interp_x = interp1d(times, x, bounds_error=False, fill_value="extrapolate")
        interp_y = interp1d(times, y, bounds_error=False, fill_value="extrapolate")
        interp_z = interp1d(times, z, bounds_error=False, fill_value="extrapolate")
        
        def position(t):
            return np.array([
                float(interp_x(t)), 
                float(interp_y(t)), 
                float(interp_z(t))
            ])
        
        return position
    
    @staticmethod
    def _create_swarm(
        initial_positions,
        avoid_radius=5.0,
        match_factor=0.1,
        centering_factor=0.01,
        noise_amp=0.2,
        flight_height=50.0,
        rng_seed=None
    ):
        """
        创建群体智能运动模型
        
        Args:
            initial_positions: 初始位置列表 [[x1,y1,z1], [x2,y2,z2], ...]
            avoid_radius: 避免碰撞半径
            match_factor: 速度匹配因子
            centering_factor: 向心力因子
            noise_amp: 随机噪声幅度
            flight_height: 飞行高度
            rng_seed: 随机数种子
        """
        rng = np.random.default_rng(rng_seed)
        num_targets = len(initial_positions)
        positions = np.array(initial_positions)
        velocities = rng.uniform(-2, 2, (num_targets, 3))
        velocities[:, 2] = rng.uniform(-0.5, 0.5, num_targets)  # 初始化z轴速度
        last_t = 0
        
        def update(t):
            nonlocal positions, velocities, last_t
            dt = t - last_t
            if dt <= 0:
                return positions.copy()
            
            new_positions = positions.copy()
            new_velocities = velocities.copy()
            
            # 群集规则 1: 避免碰撞
            for i in range(num_targets):
                for j in range(num_targets):
                    if i != j:
                        diff = positions[i] - positions[j]
                        dist = np.linalg.norm(diff)
                        if dist < avoid_radius:
                            new_velocities[i] += diff / (dist + 0.1)
            
            # 群集规则 2: 速度匹配
            avg_velocity = np.mean(new_velocities, axis=0)
            for i in range(num_targets):
                # 群集规则 3: 向中心靠拢
                center = np.mean(new_positions, axis=0)
                center[2] = flight_height  # 保持固定高度
                
                new_velocities[i] += centering_factor * (center - new_positions[i])
                
                # 速度匹配
                new_velocities[i] += match_factor * (avg_velocity - new_velocities[i])
                
                # 添加随机噪声
                new_velocities[i] += rng.normal(0, noise_amp, 3)
                new_velocities[i][2] = rng.normal(0, noise_amp*0.2)  # 限制高度变化
                
                # 限制高度范围
                if new_positions[i][2] < 0.5 * flight_height:
                    new_velocities[i][2] += 1.0
                elif new_positions[i][2] > 1.5 * flight_height:
                    new_velocities[i][2] -= 1.0
                    
                # 更新位置
                new_positions[i] += new_velocities[i] * dt
            
            positions = new_positions
            velocities = new_velocities
            last_t = t
            return positions.copy()
        
        return update

class DynamicTarget:
    """
    动态目标管理类
    
    Args:
        model_type: 运动模型类型 (MotionModelType)
        **kwargs: 模型特定的参数和属性
        
    Properties:
        position_function: 位置获取函数 f(t) -> np.array (3,)
        rcs: 雷达散射截面积
    """
    
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        
        # 提取并移除 rcs 参数
        self.rcs = kwargs.pop('rcs', 1.0)  # 默认RCS为1.0 m²
        
        # 创建位置函数
        self.position_function = MotionModelFactory.create(model_type, **kwargs)
    
    def get_position(self, t):
        """获取指定时间点的目标位置"""
        return self.position_function(t)
    
    def get_rcs(self):
        """获取目标的雷达散射截面积"""
        return self.rcs

class SwarmTargets:
    """
    群体目标管理类
    
    Args:
        num_targets: 群体中的目标数量
        model_type: 群体模型类型
        **kwargs: 模型特定的参数
        
    Properties:
        update_function: 群体更新函数 f(t) -> np.array (n,3)
        positions: 当前群体位置数组
        rcs_list: 每个目标的雷达散射截面积列表
    """
    
    def __init__(self, num_targets, model_type=MotionModelType.SWARM, **kwargs):
        self.num_targets = num_targets
        self.model_type = model_type
        
        # 处理初始位置
        if 'initial_positions' not in kwargs:
            # 自动生成初始位置
            x_center = kwargs.get('x_center', 100)
            y_center = kwargs.get('y_center', 0)
            z_center = kwargs.get('z_center', 50)
            area_size = kwargs.get('area_size', 50)
            
            positions = []
            for _ in range(num_targets):
                x = x_center + (np.random.rand() - 0.5) * area_size
                y = y_center + (np.random.rand() - 0.5) * area_size
                z = z_center + (np.random.rand() - 0.5) * area_size * 0.2
                positions.append([x, y, z])
            kwargs['initial_positions'] = positions
        
        self.update_function = MotionModelFactory.create(model_type, **kwargs)
        self.positions = self.update_function(0)
        
        # 设置RCS值
        default_rcs = kwargs.pop('rcs', 0.1)  # 提取并移除rcs
        self.rcs_list = [kwargs.get(f'rcs_{i}', default_rcs) for i in range(num_targets)]
    
    def update(self, t):
        """更新群体状态到指定时间点"""
        self.positions = self.update_function(t)
        return self.positions
    
    def get_positions(self):
        """获取当前群体位置数组"""
        return self.positions
    
    def get_rcs_list(self):
        """获取群体中每个目标的RCS列表"""
        return self.rcs_list