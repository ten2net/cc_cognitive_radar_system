from typing import Dict
import numpy as np
import radarsimpy as rp
from ..radar_system import RadarFactory, DefaultRadarFactory
from radarsimpy.simulator import sim_radar
import radarsimpy.processing as proc
from scipy.constants import speed_of_light
from scipy import signal

def normalize_rd_map(rd_map):
    """归一化距离-多普勒图"""
    # 方法1：按最大值归一化
    max_val = np.max(np.abs(rd_map))
    if max_val > 0:
        return rd_map / max_val
    
    # 方法2：按中值归一化
    median_val = np.median(np.abs(rd_map))
    if median_val > 0:
        return rd_map / median_val
    
    return rd_map

class RadarSimulator:
    """Wrapper for radar simulation using RadarSimPy"""
    
    def __init__(self, radar_type: str, params: Dict = {}):
        self.radar = DefaultRadarFactory().create(radar_type)
        self.default_radar = DefaultRadarFactory().create(radar_type)
        self.last_simulation = None
        self.last_obs = None
        
    def simulate(self, targets: list) -> np.ndarray:
        """Run radar simulation with current parameters"""
        data = sim_radar(
            self.radar,
            targets
        )
        
        self.last_simulation = data
        timestamp = data["timestamp"]
        baseband=data["baseband"]
        noise = data["noise"] 
        return baseband + noise 
    
    def process_signals(self, baseband: np.ndarray) -> np.ndarray:
        """Process raw radar signals"""
        
        # 计算每个脉冲的采样点数
        samples_per_pulse = self.radar.sample_prop["samples_per_pulse"]
        pulses = self.radar.radar_prop["transmitter"].waveform_prop["pulses"]
        
        # 创建窗函数
        # range_window = signal.windows.chebwin(self.radar.sample_prop["samples_per_pulse"], at=60)
        range_window = signal.windows.chebwin(samples_per_pulse, at=60)
        dop_window = signal.windows.chebwin(pulses, at=60)      
        
        # 进行距离FFT，使用窗函数和指定FFT点数
        self.range_data = proc.range_fft(
            baseband, 
            rwin=range_window)
        max_range = (  # 计算最大探测距离
            speed_of_light  # 光速
            * self.radar.radar_prop["receiver"].bb_prop["fs"]  # 采样率
            * self.radar.radar_prop["transmitter"].waveform_prop["pulse_length"]  # 脉冲时长 Tc
            / self.radar.radar_prop["transmitter"].waveform_prop["bandwidth"]
            / 2
        ) 
        # 多普勒FFT
        self.doppler_data =proc.doppler_fft(self.range_data, 
                                             dwin=dop_window)       
        
        # 距离-多普勒FFT
        rd_map = proc.range_doppler_fft(
            baseband, 
            rwin=range_window,
            dwin=dop_window,
            rn=samples_per_pulse,
            dn=pulses)  
        
        return normalize_rd_map(rd_map.squeeze(0))
    
    def get_observation(self, baseband: np.ndarray) -> Dict:
        """Generate observation from raw radar data"""
        rd_map = self.process_signals(baseband)
        features = self.extract_features(rd_map)
        
        self.last_obs = {
            'raw_data': baseband,
            'rd_map': rd_map,
            'features': features
        }
        
        return self.last_obs
    
    def extract_features(self, rd_map: np.ndarray) -> np.ndarray:
        """Extract features from processed radar data"""
        # Peak detection
        max_val = np.max(rd_map)
        max_idx = np.unravel_index(np.argmax(rd_map), rd_map.shape)
        
        # Statistical features
        mean_val = np.mean(rd_map)
        std_val = np.std(rd_map)
        energy = np.sum(rd_map**2)
        
        # Number of detections above threshold
        threshold = mean_val + 2 * std_val
        detections = np.sum(rd_map > threshold)
        
        return np.array([max_val, *max_idx, mean_val, std_val, energy, detections])
    

    def update_radar_params(self, params: Dict) :
        pass
        # """Update radar parameters"""
        # # Beam control
        # if 'beam_az' in params:
        #     self.radar.radar_prop['receiver'].rxchannel_prop["az_angles"] = params['beam_az']
        # if 'beam_el' in params:
        #     self.radar.radar_prop['receiver'].rxchannel_prop["el_angles"] = params['beam_el']
            
        # # Gain control
        # if 'gain' in params:
        #     self.radar.radar_prop['receiver'].rxchannel_prop["antenna_gains"] = params['gain']
            
        # # Waveform parameters
        # if 'bandwidth' in params:
        #     self.radar.radar_prop['transmitter'].waveform_prop["bandwidth"] = params['bandwidth']
        #     self.radar.radar_prop['transmitter'].waveform_prop["f"] = params['frequency']
            
        # if 'frequency' in params:
        #     self.radar.radar_prop['transmitter'].waveform_prop["f"] = params['frequency']
            
        # if 'pulse_width' in params:
        #     self.radar.radar_prop['transmitter'].waveform_prop["pulse_length"] = params['pulse_width']
            
        # if 'prf' in params:
        #     self.radar.radar_prop['transmitter'].waveform_prop["prp"] = 1 / params['prf']
            
        # if 'tx_power' in params:
        #     self.radar.radar_prop['transmitter'].rf_prop["tx_power"] = params['tx_power']        

    
    def reset_radar(self) -> None:
        """Reset radar to default parameters"""
        self.radar=self.default_radar
        self.last_simulation = None
        self.last_obs = None
        
    def randomize_radar(self) -> None:
        """Randomize radar parameters for domain randomization"""
        self.radar.radar_prop['receiver'].rf_prop["noise_figure"] = np.random.uniform(10, 15)

        self.radar.radar_prop['transmitter'].rf_prop["tx_power"] = np.random.uniform(5, 15)
        
    def get_current_radar_params(self) -> Dict:
        params ={}
        
        params['beam_az']=30 #self.radar.radar_prop['receiver'].rxchannel_prop["az_angles"]  
        params['beam_el']=5 #self.radar.radar_prop['receiver'].rxchannel_prop["el_angles"]  
        params['gain']= self.radar.radar_prop['receiver'].rxchannel_prop["antenna_gains"] 
            
        params['bandwidth']= self.radar.radar_prop['transmitter'].waveform_prop["bandwidth"] 
        params['frequency']=self.radar.radar_prop['transmitter'].waveform_prop["f"]
            
        params['pulse_width']= self.radar.radar_prop['transmitter'].waveform_prop["pulse_length"]  
        prf = 1 / self.radar.radar_prop['transmitter'].waveform_prop["prp"][0]    
        params['prf'] = prf
        params['tx_power']  =self.radar.radar_prop['transmitter'].rf_prop["tx_power"]   
             
        # 1. 基本性能
        wavelength = 3e8 / np.mean(self.radar.radar_prop['transmitter'].waveform_prop["f"])
        pulses=self.radar.radar_prop['transmitter'].waveform_prop["pulses"] 
        
        # 2. 速度分辨率
        params['velocity_resolution'] = wavelength / (2 * pulses * (1/prf))
        
        # 3. 多普勒分辨率
        params['doppler_resolution'] = 2 * params['velocity_resolution'] / wavelength
                     
        params['range_resolution']  = 3e8 / (2 * self.radar.radar_prop['transmitter'].waveform_prop["bandwidth"] )
        params['max_unambiguous_range'] = 3e8 / (2 * prf)
        params['max_unambiguous_velocity']  = prf * wavelength / 4      
    
        return params