from typing import Optional
import numpy as np
from collections import defaultdict
import time

from cognitive_radar.target.dynamic_targets import DynamicTarget, MotionModelType, SwarmTargets

class ScenarioManager:
    """
    场景管理器类，用于管理多个动态目标和群体目标的创建、更新和状态跟踪
    
    Args:
        start_time (float): 仿真开始时间（秒）
        duration (float): 仿真持续时间（秒），默认为无限运行
        time_step (float): 默认更新时间步长（秒）
    """
    
    def __init__(self, start_time: float = 0, duration: float = float('inf'), time_step: float = 0.1):
        self.start_time = start_time
        self.current_time = start_time
        self.duration = duration
        self.time_step = time_step
        
        # 存储目标计划：时间 -> [目标创建指令字典]
        self.target_schedule = defaultdict(list)
        
        # 活跃目标存储
        self.active_targets = []          # 单个目标列表
        self.active_swarms = []           # 群体目标列表
        self.target_creation_times = {}   # 目标创建时间 {目标ID: 创建时间}
        self.target_end_times = {}         # 目标结束时间 {目标ID: 结束时间}
        self.next_id = 1                  # 下一个可用的目标ID

    def schedule_target(self, create_time: float, model_type, **kwargs):
        """
        在指定时间安排创建一个目标
        
        Args:
            create_time (float): 目标创建时间（秒）
            model_type: 运动模型类型 (MotionModelType)
            **kwargs: 目标参数，包含可选参数:
                - end_time (float): 目标结束时间（秒）
                - id_prefix (str): 自定义ID前缀
                - ... 其他目标参数
        """
        # 自动添加目标ID
        target_id = f"{kwargs.get('id_prefix', 'target')}_{self.next_id}"
        self.next_id += 1
        
        # 存储创建指令
        instruction = {
            'target_id': target_id,
            'model_type': model_type,
            'create_time': create_time,
            'params': kwargs
        }
        
        self.target_schedule[create_time].append(instruction)
        
        # 如果有结束时间，记录下来
        if 'end_time' in kwargs:
            self.target_end_times[target_id] = kwargs['end_time']

    def update(self, t: Optional[float] = None):
        """
        更新所有目标状态到指定时间
        
        Args:
            t (float): 更新到的时间点（秒）。如果为None，则使用当前时间+时间步长
        """
        # 处理时间更新
        if t is not None:
            # 提供绝对时间
            self.current_time = t
        else:
            # 自动步进时间
            self.current_time += self.time_step
        
        # 检查是否有需要创建的目标
        self._create_scheduled_targets()
        
        # 检查是否有需要移除的过期目标
        self._remove_expired_targets()
        
        # 更新所有活跃目标的位置
        self._update_all_targets()

    def _create_scheduled_targets(self):
        """创建当前时间计划创建的目标"""
        # 查找在时间区间 [current_time, current_time + time_step) 内的目标
        to_create = []
        for create_time, instructions in list(self.target_schedule.items()):
            # 使用时间区间检查解决浮点数精度问题
            if create_time <= self.current_time < create_time + self.time_step:
                to_create.extend(instructions)
                del self.target_schedule[create_time]
        
        # 按照计划时间排序，确保按顺序创建
        to_create.sort(key=lambda inst: inst['create_time'])
        
        for instruction in to_create:
            self._create_target(instruction)

    def _create_target(self, instruction):
        """实际创建目标实例"""
        params = instruction['params'].copy()
        
        # 特殊处理群体目标
        if instruction['model_type'] == MotionModelType.SWARM:
            num_targets = params.pop('num_targets', 5)  # 默认5个目标
            
            # 生成初始位置
            x_center = params.pop('x_center', 200)
            y_center = params.pop('y_center', 0)
            z_center = params.pop('z_center', 50)
            area_size = params.pop('area_size', 50)
            
            initial_positions = []
            for _ in range(num_targets):
                x = x_center + (np.random.rand() - 0.5) * area_size
                y = y_center + (np.random.rand() - 0.5) * area_size
                z = z_center + (np.random.rand() - 0.5) * area_size * 0.2
                initial_positions.append([x, y, z])
            
            # 创建群体目标，只传递模型需要的参数
            swarm_params = {
                'initial_positions': initial_positions,
                'avoid_radius': params.pop('avoid_radius', 5.0),
                'match_factor': params.pop('match_factor', 0.1),
                'centering_factor': params.pop('centering_factor', 0.01),
                'noise_amp': params.pop('noise_amp', 0.2),
                'flight_height': params.pop('flight_height', 50.0),
                'rng_seed': params.pop('rng_seed', None)
            }
            
            # 创建群体目标
            swarm = SwarmTargets(num_targets, instruction['model_type'], **swarm_params)
            
            # 设置群体目标的RCS值
            swarm_rcs = params.get('rcs', 0.1)
            swarm.rcs_list = [swarm_rcs] * num_targets
            
            self.active_swarms.append({
                'id': instruction['target_id'],
                'obj': swarm,
                'create_time': self.current_time,
                'rcs': swarm_rcs  # 存储RCS值
            })
        else:
            # 移除模型不需要的参数
            model_params = params.copy()
            model_params.pop('end_time', None)
            model_params.pop('id_prefix', None)
            
            # 创建单个目标
            target = DynamicTarget(instruction['model_type'], **model_params)
            self.active_targets.append({
                'id': instruction['target_id'],
                'obj': target,
                'create_time': self.current_time,
                'rcs': params.get('rcs', 1.0)  # 存储RCS值
            })
        
        # 记录创建时间
        self.target_creation_times[instruction['target_id']] = self.current_time

    def _remove_expired_targets(self):
        """移除过期目标"""
        # 移除单个目标
        active_targets = []
        for target in self.active_targets:
            target_id = target['id']
            if target_id in self.target_end_times and self.current_time >= self.target_end_times[target_id]:
                continue  # 跳过过期目标
            active_targets.append(target)
        self.active_targets = active_targets
        
        # 移除群体目标
        active_swarms = []
        for swarm in self.active_swarms:
            swarm_id = swarm['id']
            if swarm_id in self.target_end_times and self.current_time >= self.target_end_times[swarm_id]:
                continue  # 跳过过期群体
            active_swarms.append(swarm)
        self.active_swarms = active_swarms

    def _update_all_targets(self):
        """更新所有目标到当前时间"""
        # 更新单个目标
        for target in self.active_targets:
            # 调用位置计算函数，确保模型正确更新状态
            position = target['obj'].get_position(self.current_time)
        
        # 更新群体目标
        for swarm in self.active_swarms:
            positions = swarm['obj'].update(self.current_time)

    def get_all_targets_state(self):
        """
        获取所有活动目标的状态信息（包括群体中的个体）
        
        Returns:
            list: 目标状态字典列表，每个字典包含:
                - id: 目标ID
                - type: "single"或"swarm_member"
                - position: [x, y, z] 当前位置
                - rcs: RCS值
                - model_type: 运动模型类型
        """
        states = []
        
        # 处理单个目标
        for target in self.active_targets:
            states.append({
                'id': target['id'],
                'type': 'single',
                'position': target['obj'].get_position(self.current_time),
                'rcs': target.get('rcs', 1.0),  # 使用存储的RCS值
                'model_type': str(target['obj'].model_type)
            })
        
        # 处理群体目标
        for swarm in self.active_swarms:
            swarm_id = swarm['id']
            positions = swarm['obj'].get_positions()
            rcs_list = swarm['obj'].rcs_list  # 直接访问rcs_list属性
            for i, (pos, rcs) in enumerate(zip(positions, rcs_list)):
                states.append({
                    'id': f"{swarm_id}_{i}",
                    'type': 'swarm_member',
                    'position': pos,
                    'rcs': rcs,
                    'model_type': str(swarm['obj'].model_type)
                })
        
        return states

    def get_target_count(self):
        """
        获取当前所有活动目标数量（包括群体中的个体）
        """
        count = len(self.active_targets)
        for swarm in self.active_swarms:
            count += len(swarm['obj'].get_positions())
        return count

    def clear_all_targets(self):
        """清除所有目标和计划"""
        self.target_schedule.clear()
        self.active_targets.clear()
        self.active_swarms.clear()
        self.target_creation_times.clear()
        self.target_end_times.clear()
        self.next_id = 1
        
def convert_target_states(target_states):
    """
    将场景管理器返回的目标状态转换为标准目标字典列表
    
    Args:
        target_states (list): scene.get_all_targets_state()返回的目标状态列表
        
    Returns:
        list: 标准目标字典列表，每个字典包含:
            - location: (x, y, z) 当前位置
            - speed: (vx, vy, vz) 当前速度
            - rcs: 雷达散射截面积
            - phase: 相位（默认为0）
    """
    # 这里需要实现速度计算逻辑
    # 由于原始状态只包含位置，我们需要计算速度
    # 这需要存储历史位置信息
    
    # 创建一个字典来存储历史位置
    if not hasattr(convert_target_states, "prev_positions"):
        convert_target_states.prev_positions = {} # type: ignore
        convert_target_states.prev_time = None # type: ignore
    
    current_time = time.time()
    
    # 如果没有历史时间，初始化并返回空速度
    if convert_target_states.prev_time is None: # type: ignore
        convert_target_states.prev_time = current_time # type: ignore
        convert_target_states.prev_positions = {state['id']: state['position'] for state in target_states} # type: ignore
        return [
            {
                'location': tuple(float(x) for x in state['position']),
                'speed': (0, 0, 0),  # 初始速度为0
                'rcs': state['rcs'],
                'phase': 0
            }
            for state in target_states
        ]
    
    # 计算时间差
    dt = current_time - convert_target_states.prev_time # type: ignore
    if dt <= 0:
        dt = 0.1  # 避免除以0
    
    # 转换目标状态
    converted_targets = []
    for state in target_states:
        target_id = state['id']
        current_position = tuple(float(x) for x in state['position'])
        
        # 计算速度
        if target_id in convert_target_states.prev_positions: # type: ignore
            prev_position = convert_target_states.prev_positions[target_id] # type: ignore
            # 速度 = (当前位置 - 上一位置) / 时间差
            speed = (
                float((current_position[0] - prev_position[0]) / dt),
                float((current_position[1] - prev_position[1]) / dt),
                float((current_position[2] - prev_position[2]) / dt)
            )
        else:
            speed = (0, 0, 0)  # 新目标初始速度为0
        
        converted_targets.append({
            'location': current_position,
            'speed': speed,
            'rcs': state['rcs'],
            'phase': 0  # 相位默认为0
        })
    
    # 更新历史位置和时间
    convert_target_states.prev_positions = {state['id']: state['position'] for state in target_states} # type: ignore
    convert_target_states.prev_time = current_time # type: ignore
    
    return converted_targets        
        
def main():
    # 创建场景管理器
    simulation_start = 0
    scene = ScenarioManager(start_time=simulation_start, time_step=0.1)

    # 计划创建不同类型的目标
    scene.schedule_target(
        create_time=5.0,  # 5秒后创建
        model_type=MotionModelType.HIGH_SPEED_DRONE,
        start_position=(0, 0, 100),
        end_position=(1000, 500, 150),
        cruise_speed=80,
        end_time=30.0,  # 30秒后消失
        rcs=5.0,  # 添加RCS值
        id_prefix="drone"
    )

    scene.schedule_target(
        create_time=10.0,
        model_type=MotionModelType.SWARM,
        num_targets=5,  # 减少目标数量，便于调试
        x_center=200,
        y_center=100,
        z_center=50,
        rcs=0.5,  # 设置群体目标的RCS值
        id_prefix="swarm"
    )

    # 添加额外的目标
    scene.schedule_target(
        create_time=15.0,
        model_type=MotionModelType.SINUSOIDAL,
        base_position=(0, 100, 50),
        frequency=0.5,
        amplitude=20.0,
        axis="y",
        rcs=2.0,
        id_prefix="sinusoid"
    )

    # 主仿真循环
    simulation_duration = 40  # 仿真持续40秒
    step = 0.1
    
    # 使用相对时间而非真实时间
    while scene.current_time <= simulation_duration:
        # 使用相对时间更新场景
        scene.update()  
        target_states = scene.get_all_targets_state()
        
        targets = convert_target_states(target_states)
        for target in targets:
            # print(f"目标位置: {target['location']}, 速度: {target['speed']}, RCS: {target['rcs']}") 
            pass 
        print(f"时间: {scene.current_time:.1f}秒, 目标数量: {len(targets)}")      
        # if target_states:
        #     print(f"时间: {scene.current_time:.1f}秒, 目标数量: {len(target_states)}")
        #     # 按需输出目标信息
        #     if int(scene.current_time) % 2 == 0 and scene.current_time % 1 < step:  # 每2秒输出一次
        #         print(f"========== 时间: {scene.current_time:.1f}s ==========")
        #         for state in target_states:
        #             print(f"目标 {state['id']: <15} 位置: [{state['position'][0]:>7.1f}, {state['position'][1]:>7.1f}, {state['position'][2]:>7.1f}] m, RCS: {state['rcs']} m²")
        #         print("===================================")
        # else:
        #     print(f"时间: {scene.current_time:.1f}秒, 暂无目标")
        
        # 控制仿真步长，但不影响实际时间
        time.sleep(step)  # 模拟实时步进

if __name__ == "__main__":
    main()