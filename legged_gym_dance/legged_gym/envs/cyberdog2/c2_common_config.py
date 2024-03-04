from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class CyberCommonCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 8192
        obs_base_vel = False
        obs_base_vela = False
        obs_height = False
        binarize_base_vela = False
        single_base_vel = False
        single_base_vela = False
        single_height = False
    
    class init_state(LeggedRobotCfg.init_state):
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': -45 / 57.3,     # [rad]
            'RL_thigh_joint': -45 / 57.3,   # [rad]
            'FR_thigh_joint': -45 / 57.3,     # [rad]
            'RR_thigh_joint': -45 / 57.3,   # [rad]

            'FL_calf_joint': 70 / 57.3,   # [rad]
            'RL_calf_joint': 70 / 57.3,    # [rad]
            'FR_calf_joint': 70 / 57.3,  # [rad]
            'RR_calf_joint': 70 / 57.3,    # [rad]
        }
    
    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_reduction_scale = 1.0
        ratio_delay = 0.0

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cyberdog2/urdf/cyberdog2_v2.urdf'
        name = "cyber2"
        foot_name = "foot"
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [1.0, 3.0]
        randomize_restitution = True
        randomize_base_mass = True
        added_mass_range = [-0.5, 0.5]
        randomize_com_displacement = True
        com_displacement_range = [[-0.01, 0.0, -0.01], [0.01, 0.0, 0.01]]
        randomize_joint_props = True
        joint_friction_range = [0.03, 0.08]
        joint_damping_range = [0.02, 0.06]
        
        use_dynamic_kp_scale = False
        lag_timesteps = 6
        swing_lag_timesteps = [6, 6]
        stance_lag_timesteps = [1, 1]
    
    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards_ji22_style = False
        kappa_gait_probs = 0.07
        gait_force_sigma = 100.
        gait_vel_sigma = 10.
        base_height_target = 0.30
        class scales:
            pass
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        static_friction = 0.2
        dynamic_friction = 0.2


class CyberCommonCfgPPO(LeggedRobotCfgPPO):
    use_wandb = True
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 2.5e-4
        schedule = 'fixed'