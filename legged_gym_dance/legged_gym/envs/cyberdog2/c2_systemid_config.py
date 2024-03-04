from legged_gym.envs.cyberdog2.c2_standdance_config import CyberStandDanceConfig
import numpy as np


class CyberSysidConfig(CyberStandDanceConfig):
    class init_state(CyberStandDanceConfig.init_state):
        pos = [0.0, 0.0, 0.7]
        rot = [0.,-np.sin(np.pi / 4),0.,np.cos(np.pi / 4)]

    class asset(CyberStandDanceConfig.asset):
        fix_base_link = True
    
    class control(CyberStandDanceConfig.control):
        kd_factor_range = [1.0, 1.0]
        kd_factor_range = [1.0, 1.0]
    
    class domain_rand(CyberStandDanceConfig.domain_rand):
        joint_friction_range = [0.0, 0.3]
        joint_damping_range = [0.0, 0.1]
        lag_timesteps = 6
        swing_lag_timesteps = [6, 6]
        stance_lag_timesteps = [1, 1]