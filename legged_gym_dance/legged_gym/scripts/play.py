# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import shutil
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, export_policy_as_onnx, task_registry, Logger

import numpy as np
import torch
from tqdm import tqdm
import pickle


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    low_env_cfg = env_cfg
    low_env_cfg.env.num_envs = min(low_env_cfg.env.num_envs, 5)
    low_env_cfg.env.use_fix_target = True
    low_env_cfg.env.target_cycle = 50
    low_env_cfg.record.record = RECORD_FRAMES
    low_env_cfg.record.folder = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
    low_env_cfg.terrain.curriculum = False
    low_env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.5, 0.5, 0.0]
    low_env_cfg.rewards.curriculum = False
    env_cfg.domain_rand.push_robots = False
    # env_cfg.domain_rand.added_mass_range = [-1.0, -1.0]
    env_cfg.domain_rand.com_displacement_range = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    if os.path.exists(low_env_cfg.record.folder):
        shutil.rmtree(low_env_cfg.record.folder)
    os.makedirs(low_env_cfg.record.folder, exist_ok=True)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg, is_highlevel=(args.task == "go1_highlevel"))
    low_env = env.low_level_env if args.task == "go1_highlevel" else env
    # obs = env.get_observations()
    obs, *_ = env.reset()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        # currently, both high and low level shares the same number of obs
        if hasattr(ppo_runner.alg.actor_critic, "adaptation_module"):
            input_dim = env.num_obs * env.num_history + env.num_obs * env.num_stacked_obs
        else:
            input_dim = env.num_obs
        export_policy_as_onnx(ppo_runner.alg.actor_critic, input_dim, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(low_env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 100 # number of steps before print average episode rewards
    camera_position = np.array(low_env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(low_env_cfg.viewer.lookat) - np.array(low_env_cfg.viewer.pos)
    img_idx = 0

    total_steps = int(11 * env.max_episode_length) #used to be 5
    if hasattr(low_env_cfg.control, "extended_step"):
        total_steps = int(total_steps // low_env_cfg.control.extended_step)
    
    cur_hand_pos=[]
    target_hand_pos=[]
    actions_buffer = []
    saved_data = {
        "base_position": [], "base_quaternion": [], "foot_position": [], "foot_velocity": [], "foot_ang_velocity": [],
        "foot_contact": [], "calf_contact": []}
    episode_reward_tmp = 0
    episode_length_tmp = 0
    episode_reward_buf = []
    episode_length_buf = []
    
    #for i in tqdm(range(total_steps)):
    for i in range(total_steps): 
        with torch.no_grad():
            actions = policy(obs)
        saved_data["base_position"].append(env.root_states[0, :3].cpu().numpy())
        saved_data["base_quaternion"].append(env.root_states[0, 3: 7].cpu().numpy())
        saved_data["foot_position"].append(env.foot_positions[0].cpu().numpy())
        saved_data["foot_velocity"].append(env.foot_velocities[0].cpu().numpy())
        saved_data["foot_ang_velocity"].append(env.foot_velocities_ang[0].cpu().numpy())
        saved_data["foot_contact"].append(env.contact_forces[0, env.feet_indices].cpu().numpy())
        saved_data["calf_contact"].append(env.contact_forces[0, env.calf_indices].cpu().numpy())
        # print(i, obs[env.cam_env_id, -53: -50].cpu().numpy(), obs[env.cam_env_id, -6:].cpu().numpy())
        # if i == 200:
        #     break
        # if len(actions_buffer) < 100:
        #     actions_buffer.append(actions[env.cam_env_id].detach().cpu().numpy())
        # else:
        #     actions_buffer = np.array(actions_buffer)
        #     import matplotlib.pyplot as plt
        #     fig, ax = plt.subplots(4, 3)
        #     for r in range(4):
        #         for c in range(3):
        #             ax[r][c].plot(actions_buffer[:, r * 3 + c])
        #     plt.savefig("tmp_actions.png")
        #     break
        #print("target hand pos:",obs[0,-12:-6],"cur hand pos:",obs[0,-6:])
        
        # cur_hand_pos.append(env.hand_positions[0,:])
        # target_hand_pos.append(obs[0,-6:])
        
        obs, _, rews, dones, infos = env.step(actions.detach())
        episode_reward_tmp += rews
        episode_length_tmp += torch.ones(obs.shape[0], device=obs.device)
        # if env.reset_buf[0]:
        #     print("i=",i,"reset=",env.reset_buf)
            #print("target from obs is",obs[0,-6:],"count=",env.episode_length_buf)

        if MOVE_CAMERA:
            camera_position += camera_vel * low_env.dt
            low_env.set_camera(camera_position, camera_position + camera_direction)
        # if  0 < i < stop_rew_log:
        if infos["episode"]:
            num_episodes = torch.sum(low_env.reset_buf).item()
            if num_episodes>0:
                logger.log_rewards(infos["episode"], num_episodes)
            episode_reward_buf.extend(episode_reward_tmp[low_env.reset_buf].cpu().numpy().tolist())
            episode_reward_tmp[low_env.reset_buf] = 0.
            episode_length_buf.extend(episode_length_tmp[low_env.reset_buf].cpu().numpy().tolist())
            episode_length_tmp[low_env.reset_buf] = 0.
        # elif i==stop_rew_log:
        if logger.num_episodes >= 50:
            logger.print_rewards()
            print("Mean episode reward", np.mean(episode_reward_buf), "N episodes", len(episode_reward_buf))
            print("Mean episode length", np.mean(episode_length_buf), "N episodes", len(episode_length_buf))
            break
    
    with open('cur_hand_pos.pkl','wb') as f:
        pickle.dump(cur_hand_pos, f)
    with open('target_hand_pos.pkl','wb') as f:
        pickle.dump(target_hand_pos,f)
    with open('rollout_stats.pkl', 'wb') as f:
        pickle.dump(saved_data, f)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = True
    MOVE_CAMERA = False
    args = get_args()
    play(args)
