import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

class RadarVisualizer:
    """
    雷达环境可视化器
    
    功能：
    - 显示雷达位置和波束方向
    - 显示目标位置和轨迹
    - 显示距离-多普勒图
    - 显示雷达参数状态
    
    Args:
        render_mode: 渲染模式 ('human' 或 'rgb_array')
        max_range: 最大显示距离 (米)
        fov: 雷达视场角 (度)
    """
    
    def __init__(self, render_mode='human', max_range=1000, fov=120):
        self.render_mode = render_mode
        self.max_range = max_range
        self.fov = fov
        
        # 创建图形和子图
        self.fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        
        # 3D视图
        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        self.ax_3d.set_title('3D View')
        self.ax_3d.set_xlim(-max_range, max_range)
        self.ax_3d.set_ylim(-max_range, max_range)
        self.ax_3d.set_zlim(0, max_range/2)
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        
        # 俯视图
        self.ax_top = self.fig.add_subplot(gs[0, 1])
        self.ax_top.set_title('Top View')
        self.ax_top.set_xlim(-max_range, max_range)
        self.ax_top.set_ylim(-max_range, max_range)
        self.ax_top.set_xlabel('X (m)')
        self.ax_top.set_ylabel('Y (m)')
        self.ax_top.grid(True)
        
        # 距离-多普勒图
        self.ax_rd = self.fig.add_subplot(gs[1, 0])
        self.ax_rd.set_title('Range-Doppler Map')
        self.ax_rd.set_xlabel('Range (m)')
        self.ax_rd.set_ylabel('Doppler (m/s)')
        
        # 雷达参数状态
        self.ax_params = self.fig.add_subplot(gs[1, 1])
        self.ax_params.set_title('Radar Parameters')
        self.ax_params.axis('off')
        
        # 初始化绘图元素
        self.radar_point = None
        self.beam_wedge = None
        self.target_points = []
        self.target_trajectories = []
        self.detected_points = []
        self.rd_image = None
        self.param_text = None
        
        # 动画对象
        self.animation = None
        
        plt.tight_layout()
        
    def update(self, processed_data=None, targets=None, radar_params=None, targets_in_beam=None):
        """
        更新可视化内容
        
        Args:
            processed_data: 处理后的雷达数据 (距离-多普勒图)
            targets: 目标列表 (字典列表)
            radar_params: 雷达参数字典
            targets_in_beam: 波束内的目标列表
        """
        # 清除之前的绘图元素
        self._clear_plots()
        
        # 绘制雷达位置
        self._draw_radar()
        
        # 绘制波束
        if radar_params:
            self._draw_beam(radar_params)
        
        # 绘制目标
        if targets:
            self._draw_targets(targets, targets_in_beam)
        
        # 绘制距离-多普勒图
        if processed_data is not None:
            self._draw_range_doppler(processed_data)
        
        # 显示雷达参数
        if radar_params:
            self._display_params(radar_params)
        
        # 根据渲染模式更新显示
        if self.render_mode == 'human':
            plt.pause(0.01)
    
    def _clear_plots(self):
        """清除之前的绘图元素"""
        # 清除目标点
        for point in self.target_points:
            point.remove()
        self.target_points = []
        
        # 清除轨迹
        for traj in self.target_trajectories:
            traj.remove()
        self.target_trajectories = []
        
        # 清除检测点
        for point in self.detected_points:
            point.remove()
        self.detected_points = []
        
        # 清除波束
        if self.beam_wedge:
            self.beam_wedge.remove()
            self.beam_wedge = None
        
        # 清除距离-多普勒图
        if self.rd_image:
            self.rd_image.remove()
            self.rd_image = None
        
        # 清除参数文本
        if self.param_text:
            self.param_text.remove()
            self.param_text = None
    
    def _draw_radar(self):
        """绘制雷达位置"""
        # 在3D视图中
        if not self.radar_point:
            self.radar_point = self.ax_3d.scatter([0], [0], [0], c='r', marker='o', s=50, label='Radar')
        else:
            self.radar_point._offsets3d = ([0], [0], [0])
        
        # 在俯视图中
        self.ax_top.scatter(0, 0, c='r', marker='o', s=50, label='Radar')
    
    def _draw_beam(self, radar_params):
        """绘制雷达波束"""
        beam_az = radar_params.get('beam_az', 0)
        beam_el = radar_params.get('beam_el', 0)
        beam_width = 10  # 假设波束宽度为10度
        
        # 在俯视图中绘制波束
        beam_start = np.radians(beam_az - beam_width/2)
        beam_end = np.radians(beam_az + beam_width/2)
        self.beam_wedge = Wedge((0, 0), self.max_range, 
                               np.degrees(beam_start), np.degrees(beam_end), 
                               width=self.max_range, alpha=0.2, color='blue')
        self.ax_top.add_patch(self.beam_wedge)
        
        # 在3D视图中绘制波束
        # 简化表示：绘制一个锥体
        theta = np.linspace(beam_start, beam_end, 20)
        r = np.linspace(0, self.max_range, 10)
        theta, r = np.meshgrid(theta, r)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = r * np.tan(np.radians(beam_el))
        self.ax_3d.plot_surface(x, y, z, alpha=0.2, color='blue')
    
    def _draw_targets(self, targets, targets_in_beam=None):
        """绘制目标"""
        # 提取目标位置
        positions = [t['location'] for t in targets]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        
        # 在3D视图中绘制目标
        target_points = self.ax_3d.scatter(xs, ys, zs, c='g', marker='o', s=30, label='Targets')
        self.target_points.append(target_points)
        
        # 在俯视图中绘制目标
        top_points = self.ax_top.scatter(xs, ys, c='g', marker='o', s=30, label='Targets')
        self.target_points.append(top_points)
        
        # 标记波束内的目标
        if targets_in_beam:
            in_beam_positions = [t['location'] for t in targets_in_beam]
            in_beam_xs = [p[0] for p in in_beam_positions]
            in_beam_ys = [p[1] for p in in_beam_positions]
            in_beam_zs = [p[2] for p in in_beam_positions]
            
            # 在3D视图中标记
            in_beam_points = self.ax_3d.scatter(in_beam_xs, in_beam_ys, in_beam_zs, 
                                               c='r', marker='o', s=50, label='In Beam')
            self.target_points.append(in_beam_points)
            
            # 在俯视图中标记
            top_in_beam = self.ax_top.scatter(in_beam_xs, in_beam_ys, 
                                             c='r', marker='o', s=50, label='In Beam')
            self.target_points.append(top_in_beam)
    
    def _draw_range_doppler(self, processed_data):
        """绘制距离-多普勒图"""
        # 假设processed_data是距离-多普勒矩阵
                # 转换复数数据为dB幅度
        # 加1e-10避免log(0)错误
        db_data = 20 * np.log10(np.abs(processed_data) + 1e-10)         # type: ignore
        extent = [0, self.max_range, -100, 100]  # [xmin, xmax, ymin, ymax]
        self.rd_image = self.ax_rd.imshow(
            db_data, 
            aspect='auto', 
            extent=extent,
            origin='lower',
            cmap='viridis'
        )
        self.fig.colorbar(self.rd_image, ax=self.ax_rd, label='Signal Strength')
    
    def _display_params(self, radar_params):
        """显示雷达参数"""
        param_text = (
            # f"Frequency: {radar_params.get('frequency', 0)/1e9:.2f} GHz\n"
            # f"Pulse Width: {radar_params.get('pulse_width', 0)*1e6:.2f} μs\n"
            # f"PRF: {radar_params.get('prf', 0)/1e3:.2f} kHz\n"
            # f"Gain: {radar_params.get('gain', 0):.1f} dB\n"
            f"Beam Az: {radar_params.get('beam_az', 0):.1f}°\n"
            f"Beam El: {radar_params.get('beam_el', 0):.1f}°"
        )
        
        self.param_text = self.ax_params.text(
            0.05, 0.95, param_text,
            transform=self.ax_params.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    def get_rgb_array(self):
        """获取当前帧的RGB数组"""
        self.fig.canvas.draw()
        data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def render(self):
        """渲染当前帧"""
        if self.render_mode == 'human':
           plt.pause(0.01)
           self.fig.savefig("demo.png")
        elif self.render_mode == 'rgb_array':
            return self.get_rgb_array()
    
    def close(self):
        """关闭可视化器"""
        plt.close(self.fig)
    
    def create_animation(self, update_func, frames, interval=100):
        """创建动画"""
        self.animation = FuncAnimation(
            self.fig, update_func, frames=frames,
            interval=interval, blit=False
        )
        return self.animation
      
def main():
  # 创建可视化器
  visualizer = RadarVisualizer(render_mode='human', max_range=1000)

  # 模拟雷达参数
  radar_params = {
      'frequency': 77e9,
      'pulse_width': 10e-6,
      'prf': 10e3,
      'gain': 20,
      'beam_az': 30,
      'beam_el': 5
  }

  # 模拟目标
  targets = [
      {'location': (500, 200, 100), 'rcs': 1.0, 'speed': (10, 0, 0)},
      {'location': (300, -150, 50), 'rcs': 0.5, 'speed': (-5, 5, 0)},
      {'location': (200, 100, 30), 'rcs': 2.0, 'speed': (0, 0, 0)}
  ]

  # 模拟波束内的目标
  targets_in_beam = [targets[0]]

  # 模拟距离-多普勒图
  range_bins = 100
  doppler_bins = 50
  rd_map = np.random.rand(range_bins, doppler_bins)

  # 更新可视化
  visualizer.update(
      processed_data=rd_map,
      targets=targets,
      radar_params=radar_params,
      targets_in_beam=targets_in_beam
  )

  # 渲染
  visualizer.render()
  visualizer.fig.savefig("demo.png")
  # 关闭
  visualizer.close()  
if __name__ == "__main__":
    main()
   