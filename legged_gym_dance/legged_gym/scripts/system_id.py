import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict
import os

def main(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name="cyber2_sysid")
    # env_cfg.env.num_envs = 1
    env_cfg.record.record = False
    env_cfg.record.folder = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
    env, env_cfg = task_registry.make_env(name="cyber2_sysid", args=args, env_cfg=env_cfg, is_highlevel=(args.task == "go1_highlevel"))
    env.system_id("motor_data_cyber/motor_data_cpp.pkl")

if __name__ == "__main__":
    args = get_args()
    main(args)
