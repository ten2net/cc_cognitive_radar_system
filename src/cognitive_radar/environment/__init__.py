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

__all__ = [
    'SceneModel', 'Terrain', 'Building', 'Vegetation',
    'ClutterModel', 'ConstantClutterModel', 'EmpiricalClutterModel', 'RadarClutterSimulator',
    'PropagationModel', 'FreeSpacePropagation', 'TerrainAwarePropagation', 'RadarPropagationSimulator'
]