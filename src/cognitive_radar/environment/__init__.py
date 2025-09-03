"""
环境模型包

包含:
- 场景模型 (SceneModel)
- 杂波模型 (ClutterModel, RadarClutterSimulator)
- 传播模型 (PropagationModel, RadarPropagationSimulator)
"""

from .scene import SceneModel, Terrain, Building, Vegetation
from .clutter import ClutterModel, ConstantClutterModel, EmpiricalClutterModel, RadarClutterSimulator
from .propagation import PropagationModel, FreeSpacePropagation, TerrainAwarePropagation, RadarPropagationSimulator
# from .simulate_radar import RadarSimulator
# from .radar_env import CognitiveRadarEnv
from gymnasium.envs.registration import register


# 注册环境
register(
    id='CognitiveRadar-v0',
    entry_point='cognitive_radar.environment.radar_env:CognitiveRadarEnv',
    max_episode_steps=500,
    kwargs={
        "config": {
            "radar_type": "PD-LS02",
            "max_steps": 500,
            "action_space": {
                "type": "dict",
                "dimensions": {
                    "beam_control": 2,
                    "waveform_params": 3,
                    "gain_control": 1
                }
            },
            "targets": [
                {
                    "model_type": "HIGH_SPEED_DRONE",
                    "params": {
                        "start_position": [900, 50, 50],
                        "end_position": [1000, 200, 100],
                        "cruise_speed": 30,
                        "rcs": 0.5
                    }
                }
            ]
        }
    }
)

# 可以注册多个版本的环境
register(
    id='CognitiveRadar-v1',
    entry_point='cognitive_radar.environment.radar_env:CognitiveRadarEnv',
    max_episode_steps=1000,
    kwargs={
        "config": {
            "radar_type": "PD-LS02",
            "max_steps": 1000,
            "action_space": {
                "type": "flat",
                "dimensions": 6
            },
            "targets": [
                {
                    "model_type": "HIGH_SPEED_DRONE",
                    "params": {
                        "start_position": [900, 50, 50],
                        "end_position": [1000, 200, 100],
                        "cruise_speed": 30,
                        "rcs": 0.5
                    }
                },
                {
                    "model_type": "HIGH_SPEED_DRONE",
                    "params": {
                        "start_position": [1900, 50, 50],
                        "end_position": [2000, 200, 100],
                        "cruise_speed": -30,
                        "rcs": 0.5
                    }
                }
            ]
        }
    }
)

__all__ = [
    'SceneModel', 'Terrain', 'Building', 'Vegetation',
    'ClutterModel', 'ConstantClutterModel', 'EmpiricalClutterModel', 'RadarClutterSimulator',
    'PropagationModel', 'FreeSpacePropagation', 'TerrainAwarePropagation', 'RadarPropagationSimulator'
]