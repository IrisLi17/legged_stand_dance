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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from PIL import Image as im
from PIL import ImageDraw
from tqdm import tqdm

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(0.6, 0.0, 0.0)
        _cam_quat = quat_mul(self.base_quat[0], torch.tensor([0., 0.0, np.sin(-np.pi * 2 / 4), np.cos(-np.pi * 2 / 4)], dtype=torch.float, device=self.device))
        local_transform.r = gymapi.Quat(_cam_quat[0], _cam_quat[1], _cam_quat[2], _cam_quat[3])
        body_handle = self.gym.find_actor_rigid_body_handle(self.envs[self.cam_env_id], self.actor_handles[self.cam_env_id], "base")
        assert body_handle >= 0
        self.gym.attach_camera_to_body(self.camera_handle, self.envs[self.cam_env_id], body_handle, local_transform, gymapi.FOLLOW_POSITION)

        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        self.overshoot_buf[:] = 0.
        self.q_diff_buf = torch.abs(self.default_dof_pos.to(self.device) + self.cfg.control.action_scale * actions.to(self.device) - self.dof_pos.to(self.device))    
        # if self.cfg.record.record:
        #     print(self.common_step_counter, "action", self.actions[0])
        # Simulate delay of inference thread. It will make control thread run longer.
        _switch = np.random.uniform()
        if _switch > self.cfg.control.ratio_delay:
            decimation = self.cfg.control.decimation
        else:
            decimation = np.random.randint(self.cfg.control.decimation_range[0] + 1, self.cfg.control.decimation_range[1] + 1)
        for _ in range(decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self._apply_external_foot_force()
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                            )[:, self.feet_indices, 7:10]
            self.foot_velocities_ang = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                                )[:, self.feet_indices, 10:13]
        dof=self.dof_pos[self.cam_env_id]
        self.post_physics_step()
        # if self.cfg.record.record:
        if False:
            image = self.get_camera_image()
            image = im.fromarray(image.astype(np.uint8))
            
            # draw = ImageDraw.Draw(image)
            # txt='base pos=%s'%self.base_pos[self.cam_env_id].data
            # draw.text((4,10), txt, fill='yellow')
            # txt='base quat=%s'%self.base_quat[self.cam_env_id].data
            # draw.text((4,20), txt, fill='yellow')
            # heading_vec = quat_apply_yaw(self.base_quat, self.forward_vec)
            # txt1='heading_vec=%s'%heading_vec[self.cam_env_id]
            # draw.text((4,30), txt1, fill='yellow')
            # txt2='heading=%s'%self._get_cur_heading()[self.cam_env_id]
            # draw.text((4,40), txt2, fill='yellow')
            # txt='dofs=%s'%dof.reshape((4,3)).data
            # draw.text((4,50), txt, fill='yellow')
            # # target_dofs = self.default_dof_pos + self.actions * self.cfg.control.action_scale
            # # txt='contact=%s' % self.contact_forces[self.cam_env_id, self.feet_indices].data
            # # draw.text((4, 110), txt, fill='yellow')
            # ang_vel = quat_rotate_inverse(self.base_quat, self.base_ang_vel)[self.cam_env_id]
            # txt = 'vel=%s' % ang_vel.data
            # draw.text((4, 110), txt, fill='yellow')
            
            filename = os.path.join(self.cfg.record.folder, "%d.png" % self.common_step_counter)
            image.save(filename)

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        # if self.cfg.record.record:
        #     print(self.common_step_counter, "new obs", self.obs_buf[0])
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, :3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_forward_vec[:] = quat_rotate_inverse(self.base_quat, self.forward_vec)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
        self.foot_velocities_ang = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                              )[:, self.feet_indices, 10:13]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]

        self.calf_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.calf_indices, 0:3]
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_dof_pos[:] = self.dof_pos[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if self.cfg.init_state.reset_from_buffer and self.reset_states_buffer is None:
            if self.cfg.init_state.reset_file_name and os.path.exists(self.cfg.init_state.reset_file_name):
                import pickle
                with open(self.cfg.init_state.reset_file_name, "rb") as f:
                    self.reset_states_buffer = pickle.load(f)
            else:
                self.step_to_stand()
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        if self.cfg.rewards.curriculum and (self.common_step_counter % 200 == 0):
            self.update_reward_curriculum(env_ids)
        
        # reset robot states
        self._reset_robot_states(env_ids)
        
        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_dof_pos[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.gait_indices[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.clipped_episode_sums[env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["max_command_ang"] = self.command_ranges["ang_vel_yaw"][1]
        if self.cfg.rewards.curriculum:
            for key in self.reward_scales_final:
                if self.reward_scales_final[key] < 0:
                    self.extras["episode"]["reward_cl"] = self.reward_scales[key] / self.reward_scales_final[key]
                break
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        # randomize some properties
        self.Kp_factor[env_ids] = torch.rand(len(env_ids), 12, dtype=torch.float, device=self.device) * (
            self.cfg.control.kp_factor_range[1] - self.cfg.control.kp_factor_range[0]
        ) + self.cfg.control.kp_factor_range[0]
        self.Kd_factor[env_ids] = torch.rand(len(env_ids), 12, dtype=torch.float, device=self.device) * (
            self.cfg.control.kd_factor_range[1] - self.cfg.control.kd_factor_range[0]
        ) + self.cfg.control.kd_factor_range[0]
        # if len(self.save_data_buffer['q']) >= 500 * self.cfg.control.decimation:
        #     q_achieve = np.array(self.save_data_buffer['q'][:100])
        #     q_achieve = np.concatenate([q_achieve[:, 3:6], q_achieve[:, 0:3], q_achieve[:, 9:12], q_achieve[:, 6:9]], axis=-1) * np.array([[1, -1, -1] * 4])
        #     q_des = np.array(self.save_data_buffer['q_des'][:100])
        #     q_des = np.concatenate([q_des[:, 3:6], q_des[:, 0:3], q_des[:, 9:12], q_des[:, 6:9]], axis=-1) * np.array([[1, -1, -1] * 4])
        #     projected_gravity = np.array(self.save_data_buffer['projected_gravity'][:100])
        #     import matplotlib.pyplot as plt
        #     fig, ax = plt.subplots(4, 3, figsize=(15, 9))
        #     for r in range(4):
        #         for c in range(3):
        #             ax[r][c].plot(q_des[:, 3 * r + c], label="q_des")
        #             ax[r][c].plot(q_achieve[:, 3 * r + c], label="achieve")
        #             if r == 3 and c == 2:
        #                 ax[r][c].legend()
        #     plt.savefig("tmp_sim_q.png")
        #     plt.close(fig)
        #     fig, ax = plt.subplots(1, 3, figsize=(10, 3))
        #     for i in range(3):
        #         ax[i].plot(projected_gravity[:, i])
        #     plt.savefig("tmp_sim_gravity.png")
        #     plt.close(fig)
        #     exit()
        for i in range(self.lag_buffer.shape[2]):
            if self.cfg.control.action_mode == "bias":
                self.lag_buffer[env_ids, :, i] = (self.dof_pos[env_ids] - self.default_dof_pos.unsqueeze(dim=0)) / self.cfg.control.action_scale
            else:
                self.lag_buffer[env_ids, :, i] = ((self.dof_pos[env_ids] - self.dof_pos_hard_limits[:, 0].unsqueeze(dim=0)) / (
                    self.dof_pos_hard_limits[:, 1] - self.dof_pos_hard_limits[:, 0]).unsqueeze(dim=0) * 2 - 1) / self.cfg.control.action_scale

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.rew_buf_pos += rew * (rew > 0)
            self.rew_buf_neg += rew * (rew < 0)
            # if torch.sum(rew) > 0:
            #     self.rew_buf_pos += rew
            # else:
            #     self.rew_buf_neg += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        elif self.cfg.rewards.only_positive_rewards_ji22_style: #TODO: update
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        self.clipped_episode_sums += self.rew_buf

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def step_to_stand(self):
        self.reset_states_buffer = {"root_states": [], "dof_pos": []}
        step_counter = 0
        for trial in tqdm(range(300)):
            self._reset_dofs_rand(torch.arange(self.num_envs, device=self.device))
            self._reset_root_states_rand(torch.arange(self.num_envs, device=self.device))
            init_dof_pos = self.dof_pos.clone()
            env_ids = torch.arange(self.num_envs, device=self.device)
            for step_idx in range(30):
                for _ in range(self.cfg.control.decimation):
                    torques = 60*(init_dof_pos - self.dof_pos) - 2*self.dof_vel
                    self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
                    self.gym.simulate(self.sim)
                    if self.device == 'cpu':
                        self.gym.fetch_results(self.sim, True)
                    self.gym.refresh_dof_state_tensor(self.sim)
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_net_contact_force_tensor(self.sim)
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.gym.refresh_jacobian_tensors(self.sim)
                foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
                # print("step idx", step_idx, "foot height", foot_positions[:, :, 2])
                # image = self.get_camera_image()
                # image = im.fromarray(image.astype(np.uint8))
                # filename = os.path.join(self.cfg.record.folder, "%d.png" % (step_counter))
                # image.save(filename)
                step_counter += 1
                if torch.all(foot_positions[:, :, 2] < 0.02) and torch.all(torch.norm(self.root_states[:, 7:13], dim=-1) < 1):
                    break
            valid_idx = (torch.logical_and(
                torch.all(foot_positions[:, :, 2] < 0.02, dim=-1), 
                torch.norm(self.root_states[:, 7:13], dim=-1) < 1)
            ).nonzero(as_tuple=False).flatten()
            if len(valid_idx):
                self.reset_states_buffer["root_states"].append(self.root_states.clone()[valid_idx])
                self.reset_states_buffer["dof_pos"].append(self.dof_pos.clone()[valid_idx])
        self.reset_states_buffer["root_states"] = torch.cat(self.reset_states_buffer["root_states"], dim=0)
        self.reset_states_buffer["dof_pos"] = torch.cat(self.reset_states_buffer["dof_pos"], dim=0)
        print("total valid", len(self.reset_states_buffer["root_states"]))
        if self.cfg.init_state.reset_file_name:
            import pickle
            with open(self.cfg.init_state.reset_file_name, "wb") as f:
                pickle.dump(self.reset_states_buffer, f)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                # num_buckets = 64
                # bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                # friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                # self.friction_coeffs = friction_buckets[bucket_ids]
                self.friction_coeffs = torch.rand(
                    size=(self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False
                ) * (friction_range[1] - friction_range[0]) + friction_range[0]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        else:
            if env_id == 0:
                self.friction_coeffs = torch.ones((self.num_envs,), dtype=torch.float, device=self.device) * props[0].friction
        if env_id == self.cam_env_id:
            print("friction")
            for s in range(len(props)):
                print(s, props[s].friction)
        if self.cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch.rand(
                    size=(self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False
                ) * (restitution_range[1] - restitution_range[0]) + restitution_range[0]
            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
        else:
            if env_id == 0:
                self.restitution_coeffs = torch.ones((self.num_envs,), dtype=torch.float, device=self.device) * props[0].restitution
        if env_id == self.cam_env_id:
            print("restitution")
            for s in range(len(props)):
                print(s, props[s].restitution)
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        props = props.copy()
        if env_id==0:
            # hasLimits, lower, upper, driveMode, velocity, effort, stiffness, damping, friction, armature
            # True, x, x, 3, x, x, 0., 0.01, 0.1, 0.
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_pos_hard_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.Kp_factor = torch.rand(self.num_envs, 12, dtype=torch.float, device=self.device) * (
                self.cfg.control.kp_factor_range[1] - self.cfg.control.kp_factor_range[0]
            ) + self.cfg.control.kp_factor_range[0]
            self.Kd_factor = torch.rand(self.num_envs, 12, dtype=torch.float, device=self.device) * (
                self.cfg.control.kd_factor_range[1] - self.cfg.control.kd_factor_range[0]
            ) + self.cfg.control.kd_factor_range[0]
            for i in range(len(props)):
                # self.dof_pos_limits[i, 0] = props["lower"][i].item()
                # self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_pos_hard_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_hard_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                if self.cfg.rewards.soft_dof_pos_low is not None:
                    m = (self.cfg.rewards.soft_dof_pos_low[i] + self.cfg.rewards.soft_dof_pos_high[i]) / 2
                    r = self.cfg.rewards.soft_dof_pos_high[i] - self.cfg.rewards.soft_dof_pos_low[i]
                else:
                    m = (self.dof_pos_hard_limits[i, 0] + self.dof_pos_hard_limits[i, 1]) / 2
                    r = self.dof_pos_hard_limits[i, 1] - self.dof_pos_hard_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            mass_offset = np.random.uniform(rng[0], rng[1])
            self.mass_offset.append(mass_offset)
            if env_id == self.cam_env_id:
                print("mass offset", mass_offset)
            # ratio = 1 + mass_offset / props[0].mass
            # props[0].inertia.x = gymapi.Vec3(ratio * props[0].inertia.x.x, ratio * props[0].inertia.x.y, ratio * props[0].inertia.x.z)
            # props[0].inertia.y = gymapi.Vec3(ratio * props[0].inertia.y.x, ratio * props[0].inertia.y.y, ratio * props[0].inertia.y.z)
            # props[0].inertia.z = gymapi.Vec3(ratio * props[0].inertia.z.x, ratio * props[0].inertia.z.y, ratio * props[0].inertia.z.z)
            # props[0].invInertia.x = gymapi.Vec3(1/ratio * props[0].invInertia.x.x, 1/ratio * props[0].invInertia.x.y, 1/ratio * props[0].invInertia.x.z)
            # props[0].invInertia.y = gymapi.Vec3(1/ratio * props[0].invInertia.y.x, 1/ratio * props[0].invInertia.y.y, 1/ratio * props[0].invInertia.y.z)
            # props[0].invInertia.z = gymapi.Vec3(1/ratio * props[0].invInertia.z.x, 1/ratio * props[0].invInertia.z.y, 1/ratio * props[0].invInertia.z.z)
            props[0].mass += mass_offset
            # props[0].invMass = 1 / props[0].mass
        if self.cfg.domain_rand.randomize_com_displacement:
            rng = self.cfg.domain_rand.com_displacement_range
            com_displacement = np.array([np.random.uniform(rng[0][0], rng[1][0]), np.random.uniform(rng[0][1], rng[1][1]), np.random.uniform(rng[0][2], rng[1][2])])
            self.com_displacement.append(com_displacement)
            if env_id == self.cam_env_id:
                print("com displacement", com_displacement)
            props[0].com += gymapi.Vec3(
                com_displacement[0], com_displacement[1], com_displacement[2]
            )
        # TODO: seems a critical factor
        hip_mass_offset = np.random.uniform(0.0, 0.1)
        thigh_mass_offset = np.random.uniform(max(-0.05, -0.5 * props[2].mass), 0.05)
        calf_mass_offset = np.random.uniform(max(-0.05, -0.5 * props[3].mass), 0.05)
        foot_mass_offset = np.random.uniform(max(-0.01, -0.5 * props[4].mass), 0.01)
        for i in range(1, len(props)):
            if self.cfg.domain_rand.randomize_foot_mass and i % 4 == 0:
                # otherwise, there are strange collisions on calf
                props[i].mass += foot_mass_offset
                if env_id == self.cam_env_id:
                    print("foot mass offset", foot_mass_offset)
            elif self.cfg.domain_rand.randomize_hip_mass and i % 4 == 1:
                props[i].mass += hip_mass_offset
                if env_id == self.cam_env_id:
                    print("hip mass offset", hip_mass_offset)
            elif self.cfg.domain_rand.randomize_thigh_mass and i % 4 == 2:
                props[i].mass += thigh_mass_offset
                if env_id == self.cam_env_id:
                    print("thigh mass offset", thigh_mass_offset)
            elif self.cfg.domain_rand.randomize_calf_mass and i % 4 == 3:
                props[i].mass += calf_mass_offset
                if env_id == self.cam_env_id:
                    print("calf mass offset", calf_mass_offset)
        # print(props[0].com, props[0].flags, props[0].inertia, props[0].invInertia, 
        #       props[0].invMass, props[0].mass)
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._step_contact_targets()
        if self.cfg.commands.heading_command:
            self._recompute_ang_vel()

        # if self.cfg.terrain.measure_heights:
        self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # TODO: discretize bin
        sample_mode = "continuous"
        if sample_mode == "discrete":
            self.commands[env_ids, 0] = self._sample_fn(
                "discrete", [self.command_ranges["limit_vel_x"][0], self.command_ranges["limit_vel_x"][1], self.cfg.commands.num_bins_vel_x],
                (len(env_ids),), self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1])
            self.commands[env_ids, 1] = self._sample_fn(
                "discrete", [self.command_ranges["limit_vel_y"][0], self.command_ranges["limit_vel_y"][1], self.cfg.commands.num_bins_vel_y],
                (len(env_ids),), self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1]
            )
            assert not self.cfg.commands.heading_command
            self.commands[env_ids, 2] = self._sample_fn(
                "discrete", [self.command_ranges["limit_ang_vel_yaw"][0], self.command_ranges["limit_ang_vel_yaw"][1], self.cfg.commands.num_bins_ang_vel],
                (len(env_ids),), self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1]
            )
            if self.cfg.commands.num_commands >= 7:
                self.commands[env_ids, 4] = self._sample_fn(
                    "discrete", [*self.command_ranges["base_height"], self.cfg.commands.num_bins_base_height], (len(env_ids),)
                )
                
        # self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 0] = self._sample_fn("continuous", [self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1]], (len(env_ids),))
        # self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = self._sample_fn("continuous", [self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1]], (len(env_ids),))
        if self.cfg.commands.separate_lin_ang:
            mask = torch.rand_like(self.commands[env_ids, 0]) > 0.5
            self.commands[env_ids[mask], 0:2] = 0.
        if self.cfg.commands.heading_command:
            # relatively sample
            cur_heading = self._get_cur_heading()[env_ids]
            self.commands[env_ids, 3] = cur_heading + torch_rand_float(
                self.cfg.commands.ranges.heading[0], self.cfg.commands.ranges.heading[1], 
                (len(env_ids), 1), device=self.device).squeeze(1)
            if self.cfg.commands.separate_lin_ang:
                self.commands[env_ids[~mask], 3] = cur_heading[~mask]
            # absolute sample
            # self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            if self.cfg.commands.separate_lin_ang:
                self.commands[env_ids[~mask], 3] = 0.

        if self.cfg.commands.num_commands >= 7:
            # TODO: special zero frequency and zero speed: no walking?
            if self.cfg.commands.random_gait:
                self.commands[env_ids, 4] = torch_rand_float(self.command_ranges["base_height"][0], self.command_ranges["base_height"][1], (len(env_ids), 1), device=self.device).squeeze(1)
                self.commands[env_ids, 5] = torch_rand_float(self.command_ranges["foot_height"][0], self.command_ranges["foot_height"][1], (len(env_ids), 1), device=self.device).squeeze(1)
                self.commands[env_ids, 6] = torch_rand_float(self.command_ranges["frequency"][0], self.command_ranges["frequency"][1], (len(env_ids), 1), device=self.device).squeeze(1)            
            else:
                # only in initial state
                if torch.norm(self.commands[env_ids, 4: 7]).item() < 1e-3:
                    self.commands[env_ids, 4] = (self.command_ranges["base_height"][0] + self.command_ranges["base_height"][1]) / 2
                    self.commands[env_ids, 5] = (self.command_ranges["foot_height"][0] + self.command_ranges["foot_height"][1]) / 2
                    self.commands[env_ids, 6] = (self.command_ranges["frequency"][0] + self.command_ranges["frequency"][1]) / 2
        if self.cfg.commands.num_commands >= 10:
            self.commands[env_ids, 7] = torch_rand_float(0, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 8] = torch_rand_float(0, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 9] = torch_rand_float(0, 1, (len(env_ids), 1), device=self.device).squeeze(1)
            mode = torch.randint(0, 4, (len(env_ids),), device=self.device)
            mode *= 0 # force trotting
            # 0: trotting, 1: bounding, 2: pacing, 3: pronking
            trot_idx = (mode == 0).nonzero(as_tuple=False).flatten()
            bound_idx = (mode == 1).nonzero(as_tuple=False).flatten()
            pace_idx = (mode == 2).nonzero(as_tuple=False).flatten()
            pronk_idx =  (mode == 3).nonzero(as_tuple=False).flatten()
            
            self.commands[env_ids[trot_idx], 7] = 0.5
            self.commands[env_ids[trot_idx], 8] = 0
            self.commands[env_ids[trot_idx], 9] = 0

            self.commands[env_ids[bound_idx], 7] = 0
            self.commands[env_ids[bound_idx], 8] = 0
            self.commands[env_ids[bound_idx], 9] = 0.5
            
            self.commands[env_ids[pace_idx], 7] = 0
            self.commands[env_ids[pace_idx], 8] = 0.5
            self.commands[env_ids[pace_idx], 9] = 0
            
            self.commands[env_ids[pronk_idx], 7] = 0
            self.commands[env_ids[pronk_idx], 8] = 0
            self.commands[env_ids[pronk_idx], 9] = 0
        if self.cfg.commands.num_commands >= 12:
            self.commands[env_ids, 10] = torch_rand_float(self.command_ranges["stance_width"][0], self.command_ranges["stance_width"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 11] = torch_rand_float(self.command_ranges["stance_length"][0], self.command_ranges["stance_length"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.zero_cmd_threshold).unsqueeze(1)
        if self.cfg.record.record:
            # pass
            if self.cam_env_id in env_ids:
                if len(self.all_commands):
                    self.commands[env_ids, :4] = self.all_commands.pop()
                else:
                    pass
                    # self.commands[env_ids, :4] = 0.
            if self.cfg.commands.num_commands > 4:
                self.commands[env_ids, 4:5] = 0.35
                self.commands[env_ids, 5: 6] = self.command_ranges["foot_height"][1]
                self.commands[env_ids, 6: 7] = 2.5
            if len(env_ids) and self.cam_env_id in env_ids:
                print(self.commands[self.cam_env_id])
            # self.commands[env_ids] = 0
            # self.commands[env_ids, 0] = 0.6

    def _sample_fn(self, mode: str, support: list, shape, low: float = None, high: float = None):
        """
        mode: continuous or discrete
        support: [low, high] if `mode` is `continuous`, [limit low, limit, hight, num bins] if `mode` is `discrete`
        shape:
        low: additional range for discrete distribution
        high: additional range for discrete distribution
        """
        if mode == "continuous":
            return torch.rand(shape, dtype=torch.float, device=self.device) * (support[1] - support[0]) + support[0]
        elif mode == "discrete":
            candidates = np.linspace(support[0], support[1], support[2])
            if low is None:
                low = support[0]
            if high is None:
                high = support[1]
            candidates = candidates[np.logical_and(candidates >= low, candidates <= high)]
            assert len(candidates) > 0
            return torch.from_numpy(np.random.choice(candidates, size=shape), dtype=torch.float).to(self.device)
        
    def _step_contact_targets(self):
        # TODO: fill in reasonable numbers
        # frequencies = 3 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.commands.num_commands >= 10: #false here
            frequencies = self.commands[:, 6]
            phases = self.commands[:, 7]
            offsets = self.commands[:, 8]
            bounds = self.commands[:, 9]
        else:
            frequencies = self.cfg.commands.default_gait_freq # 3.57#2.5
            phases = 0.5
            offsets = 0
            bounds = 0
        
        # if mode == "trotting":
        #     phases = 0.5 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # [0, 1]
        #     offsets = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # [0, 1]
        #     bounds = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # [0, 1]
        # elif mode == "bounding":
        #     phases = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        #     offsets = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        #     bounds = 0.5 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # elif mode == "pacing":
        #     phases = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        #     offsets = 0.5 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        #     bounds = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # elif mode == "pronking":
        #     phases = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        #     offsets = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        #     bounds = 0 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # else:
        #     raise NotImplementedError
        durations = 0.5 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # 0.5
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        # TODO: 
        # if self.cfg.commands.pacing_offset:
        #     foot_indices = [self.gait_indices + phases + offsets + bounds,
        #                     self.gait_indices + bounds,
        #                     self.gait_indices + offsets,
        #                     self.gait_indices + phases]
        # else:
        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]
        
        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)
        #print("self.foot_indices=",self.foot_indices[0][:2])
        #print("gait_indices=",self.gait_indices[0],"dt,freq=",self.dt,frequencies,self.dt*frequencies)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                        0.5 / (1 - durations[swing_idxs]))

        # if self.cfg.commands.durations_warp_clock_inputs:

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        # self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
        # self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
        # self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
        # self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

        # self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
        # self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
        # self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
        # self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

        # von mises distribution
        if hasattr(self.cfg.rewards, "kappa_gait_probs"):
            #print("kappa aaaaaaaaaaaaa")
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                    kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                        smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                                1 - smoothing_cdf_start(
                                            torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
            smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                        smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                                1 - smoothing_cdf_start(
                                            torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                        smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                                1 - smoothing_cdf_start(
                                            torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                        smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                                1 - smoothing_cdf_start(
                                            torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR
            #print("desired contact state=",self.desired_contact_states[0, -2:])

        # if self.cfg.commands.num_commands > 9:
        #     self.desired_footswing_height = self.commands[:, 9]
    
    def _recompute_ang_vel(self):
        heading = self._get_cur_heading()
        self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
    
    def _get_cur_heading(self):
        heading_vec = quat_apply_yaw(self.base_quat, self.forward_vec)
        heading = torch.atan2(heading_vec[:, 1], heading_vec[:, 0])
        return heading
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # update_mask = torch.rand(self.num_envs, 1).to(self.device) > 0.0 # communication stucked rate
        # update_mask = torch.logical_or(update_mask, torch.norm(self.lagged_actions, dim=-1, keepdim=True) < 1e-3)
        # updated_actions = actions * update_mask.float() + self.lagged_actions * (1 - update_mask.float())
        updated_actions = actions
        # self.lag_buffer = self.lag_buffer[1:] + [updated_actions.clone()]
        # used_actions = self.lag_buffer[0]
        
        # According to swing or stance leg, use different lag timesteps, and update valid_history_length
        self.lag_buffer = torch.cat([self.lag_buffer[:, :, 1:], updated_actions.unsqueeze(dim=-1).clone()], dim=-1)
        terrain_at_foot_height = self._get_heights_at_points(self.foot_positions[:, :, :2])
        is_swing = torch.tile((self.foot_positions[:, :, 2] > terrain_at_foot_height + 0.019).unsqueeze(dim=-1), (1, 1, 3)).reshape((self.num_envs, self.num_dof))
        dynamic_factor = torch.ones_like(self.dof_pos)
        if self.cfg.domain_rand.use_dynamic_kp_scale:
            dynamic_factor[is_swing] = 0.85
        self.lag_steps[:] = torch.from_numpy(np.random.randint(
            low=self.cfg.domain_rand.stance_lag_timesteps[0], 
            high=self.cfg.domain_rand.stance_lag_timesteps[1] + 1,
            size=self.lag_steps.shape
        )).to(self.device)
        self.lag_steps[is_swing] = torch.from_numpy(np.random.randint(
            low=self.cfg.domain_rand.swing_lag_timesteps[0], 
            high=self.cfg.domain_rand.swing_lag_timesteps[1] + 1,
            size=self.lag_steps[is_swing].shape
        )).to(self.device) # large delay when swing
        cur_lag_steps = torch.minimum(self.valid_history_length, self.lag_steps) # min: 0, no delay; max: cfg.lag_timesteps
        # index: out[i][j] = input[i][j][index[i][j]]
        used_actions = torch.gather(self.lag_buffer, 2, (self.cfg.domain_rand.lag_timesteps - cur_lag_steps).unsqueeze(2)).squeeze(-1)
        self.valid_history_length = torch.clamp(cur_lag_steps + 1, max=self.cfg.domain_rand.lag_timesteps)
        
        # self.lagged_actions[:] = updated_actions[:]
        #pd controller
        actions_scaled = used_actions * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_reduction_scale
        # TODO: simulate delay of control thread. Maintain old actions, update with new actions with some probablity
        control_type = self.cfg.control.control_type
        if control_type=="P":
            if self.cfg.control.action_mode == "bias":
                motor_target = actions_scaled + self.default_dof_pos
            elif self.cfg.control.action_mode == "nobias":
                motor_target = self.dof_pos_hard_limits[:, 0].unsqueeze(dim=0) + (
                    torch.clamp(actions_scaled, -1, 1) + 1) / 2 * ((self.dof_pos_hard_limits[:, 1] - self.dof_pos_hard_limits[:, 0]).unsqueeze(dim=0))
            else:
                raise NotImplementedError
            self.overshoot_buf = self.overshoot_buf + (motor_target - self.dof_pos_hard_limits[:, 1].unsqueeze(dim=0)).clip(min=0) + (self.dof_pos_hard_limits[:, 0].unsqueeze(dim=0) - motor_target).clip(min=0)
            # self.q_diff_buf = self.q_diff_buf + (torch.abs(motor_target - self.dof_pos) - 2 * self.torque_limits / self.p_gains).clip(min=0)
            # motor_target = self.dof_pos + torch.clamp(motor_target - self.dof_pos, -2 * self.torque_limits / self.p_gains, 2 * self.torque_limits / self.p_gains)
            torques = self.p_gains*self.Kp_factor*dynamic_factor*(motor_target - self.dof_pos) - self.d_gains*self.Kd_factor*dynamic_factor*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        res = torch.clip(torques, -self.torque_limits * self.cfg.control.torque_scale, self.torque_limits * self.cfg.control.torque_scale)
        # self.torque_buffer.append(res[0].cpu().numpy())
        # self.save_data_buffer['q'].append(self.dof_pos[0].cpu().numpy())
        # self.save_data_buffer['q_des'].append(motor_target[0].cpu().numpy())
        # self.save_data_buffer['projected_gravity'].append(quat_rotate_inverse(self.base_quat, self.gravity_vec)[0].cpu().numpy())
        return res

    def _apply_external_foot_force(self):
        return

    def _reset_robot_states(self, env_ids):
        if self.cfg.init_state.reset_from_buffer:
            selected_idx = torch.randint(0, len(self.reset_states_buffer["dof_pos"]), size=(len(env_ids),), device=self.device)
            self.dof_pos[env_ids] = self.reset_states_buffer["dof_pos"][selected_idx]
            self.dof_vel[env_ids] = 0.

            env_ids_int32 = self.num_actors * env_ids.to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            self.root_states[env_ids] = self.reset_states_buffer["root_states"][selected_idx]
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.root_states),
                                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        else:
            self._reset_dofs_rand(env_ids)
            self._reset_root_states_rand(env_ids)

    def _reset_dofs_rand(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if not self.cfg.init_state.randomize_rot:
            self.dof_pos[env_ids] = self.init_dof_pos + torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        else:
            self.dof_pos[env_ids] = self.dof_pos_hard_limits[:, 0] + (
                self.dof_pos_hard_limits[:, 1] - self.dof_pos_hard_limits[:, 0]) * torch_rand_float(
                    0.0, 1.0, (len(env_ids), self.num_dof), device=self.device
                )
        self.dof_vel[env_ids] = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        # TODO: Important! should feed actor id, not env id
        env_ids_int32 = self.num_actors * env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_rand(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if self.cfg.init_state.randomize_rot:
            rand_rpy = torch_rand_float(-np.pi, np.pi, (len(env_ids), 3), device=self.device)
            self.root_states[env_ids, 3: 7] = quat_from_euler_xyz(rand_rpy[:, 0], rand_rpy[:, 1], rand_rpy[:, 2])
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.1, 0.1, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.all_root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
        # if torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_ang_vel"]:
        #     self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 0.5, -self.cfg.commands.max_ang_vel_yaw, 0.)
        #     self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 0.5, 0., self.cfg.commands.max_ang_vel_yaw)

    def update_reward_curriculum(self, env_ids):
        metric = torch.mean(self.clipped_episode_sums[env_ids])
        print("reward metric", metric)
        if metric > 0.2:
            for key in self.reward_scales:
                if self.reward_scales_final[key] < 0:
                    scale = max(self.reward_scales[key] + self.cfg.rewards.cl_step * self.reward_scales_final[key], self.reward_scales_final[key])
                    self.reward_scales[key] = scale
            
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        # create some wrapper tensors for different slices
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self.all_root_states.view(self.num_envs, -1, 13)[:, 0, :]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_pos = self.root_states[:, :3]
        self.base_quat = self.root_states[:, 3:7]

        # self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]
        self.lag_buffer = torch.zeros((self.num_envs, self.num_dof, self.cfg.domain_rand.lag_timesteps + 1), dtype=torch.float, device=self.device)
        self.lag_steps = torch.zeros((self.num_envs, self.num_dof), dtype=int, device=self.device)
        self.valid_history_length = torch.ones((self.num_envs, self.num_dof), dtype=int, device=self.device) * (self.cfg.domain_rand.lag_timesteps + 1)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        self.jacobian = gymtorch.wrap_tensor(_jacobian)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        # TODO: only for cyber dog
        if hasattr(self.cfg.rewards, "upright_vec"):
            self.upright_vec = to_torch(self.cfg.rewards.upright_vec, device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.lagged_actions = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        if self.cfg.commands.num_commands > 4:
            self.commands_gait_mean = torch.tensor([
                np.mean(self.command_ranges["base_height"]), 
                np.mean(self.command_ranges["foot_height"]),
                np.mean(self.command_ranges["frequency"]),
                0., 0., 0.
            ], device=self.device, requires_grad=False).float()
            self.commands_gait_scale = torch.tensor([
                (self.command_ranges["base_height"][1] - self.command_ranges["base_height"][0]) / 2,
                (self.command_ranges["foot_height"][1] - self.command_ranges["foot_height"][0]) / 2,
                (self.command_ranges["frequency"][1] - self.command_ranges["frequency"][0]) / 2,
                1., 1., 1.
            ], device=self.device, requires_grad=False).float()
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_forward_vec = quat_rotate_inverse(self.base_quat, self.forward_vec)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]
        self.foot_velocities_ang = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 10: 13]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.calf_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.calf_indices, 0:3]
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # if self.cfg.terrain.measure_heights:
        self.height_points = self._init_height_points()
        self.measured_heights = 0

        self.reset_states_buffer = None
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.init_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.init_dof_pos_range = torch.zeros((self.num_dof, 2), dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            init_angle = self.cfg.init_state.init_joint_angles[name]
            self.init_dof_pos[i] = init_angle
            self.init_dof_pos_range[i][0] = self.cfg.init_state.init_joint_angles_range[name][0]
            self.init_dof_pos_range[i][1] = self.cfg.init_state.init_joint_angles_range[name][1]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.init_dof_pos = self.init_dof_pos.unsqueeze(0)
        self.overshoot_buf = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.q_diff_buf = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) # motor target and cur q
        if hasattr(self.cfg.init_state, "HIP_OFFSETS"):
            self.HIP_OFFSETS = torch.from_numpy(self.cfg.init_state.HIP_OFFSETS).to(dtype=torch.float, device=self.device)  # (4, 3)
        if hasattr(self.cfg.init_state, "DEFAULT_HIP_POSITIONS"):
            self.DEFAULT_HIP_POSITIONS = torch.from_numpy(np.array(self.cfg.init_state.DEFAULT_HIP_POSITIONS)).to(dtype=torch.float, device=self.device)
        self.save_data_buffer = {"q": [], "q_des": [], "projected_gravity": []}
        if self.cfg.record.record:
            if self.cfg.commands.heading_command:
                self.all_commands = [
                    torch.from_numpy(np.array([0., self.command_ranges["lin_vel_y"][1], 0., 0.])).float().to(self.device),
                    torch.from_numpy(np.array([self.command_ranges["lin_vel_x"][0], 0., 0., 0.])).float().to(self.device),
                    torch.from_numpy(np.array([self.command_ranges["lin_vel_x"][1], 0., 0., 0.])).float().to(self.device),
                    torch.from_numpy(np.array([0., 0., 0., -np.pi / 2])).float().to(self.device),
                    torch.from_numpy(np.array([0., 0., 0., np.pi / 2])).float().to(self.device),
                    torch.from_numpy(np.array([self.command_ranges["lin_vel_x"][1], 0., 0., 0.])).float().to(self.device),
                    torch.from_numpy(np.array([self.command_ranges["lin_vel_x"][0], 0., 0., 0.])).float().to(self.device),
                    # torch.from_numpy(np.array([0., 0., 0., np.pi])).float().to(self.device),
                ]
                self.all_commands = []
            else:
                self.all_commands = [
                    torch.from_numpy(np.array([self.command_ranges["lin_vel_x"][0], 0., 0., 0.])).float().to(self.device),
                    torch.from_numpy(np.array([self.command_ranges["lin_vel_x"][1], 0., 0., 0.])).float().to(self.device),
                    torch.from_numpy(np.array([0., 0., self.command_ranges["ang_vel_yaw"][0], 0.])).float().to(self.device),
                    torch.from_numpy(np.array([0., 0., self.command_ranges["ang_vel_yaw"][1], 0.])).float().to(self.device),
                    torch.from_numpy(np.array([0., 0., self.command_ranges["ang_vel_yaw"][1], 0.])).float().to(self.device),
                    torch.from_numpy(np.array([0., 0., self.command_ranges["ang_vel_yaw"][1], 0.])).float().to(self.device),
                    torch.from_numpy(np.array([0., 0., self.command_ranges["ang_vel_yaw"][1], 0.])).float().to(self.device),
                ]

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        self.clipped_episode_sums = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        if hasattr(self.cfg.rewards, "curriculum") and self.cfg.rewards.curriculum:
            self.reward_scales_final = self.reward_scales.copy()
            for key in self.reward_scales:
                if self.reward_scales_final[key] < 0:
                    self.reward_scales[key] *= self.cfg.rewards.cl_init
    
    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border_size 
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)  
        # 900, 900
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s] # FL, FR, RL, RR
        calf_names = [s for s in body_names if "calf" in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name == s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        allow_initial_contact_names = []
        for name in self.cfg.asset.allow_initial_contacts_on:
            allow_initial_contact_names.extend([s for s in body_names if name in s])
        
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.num_actors = 1
        self.joint_friction = []
        self.joint_damping_range = self.cfg.domain_rand.joint_damping_range
        self.joint_friction_range = self.cfg.domain_rand.joint_friction_range
        self.com_displacement = []
        self.mass_offset = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            # randomize?
            for j in range(len(dof_props)):
                if self.cfg.domain_rand.randomize_joint_props:
                    dof_props["damping"][j] = np.random.uniform(*self.joint_damping_range)
                    dof_props["friction"][j] = np.random.uniform(*self.joint_friction_range)
                    # if self.cfg.record.record:
                    #     dof_props["damping"][j] = 0.01
                    #     dof_props["friction"][j] = 0.1
            if i == self.cam_env_id:
                print("joint props")
                for j in range(len(dof_props)):
                    print(j, "damping", dof_props["damping"][j], "friction", dof_props["friction"][j])
            self.joint_friction.append(np.array([dof_props["friction"][j] for j in range(len(dof_props))]))
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.joint_friction = torch.from_numpy(np.array(self.joint_friction)).float().to(self.device)
        if self.cfg.domain_rand.randomize_base_mass:
            self.mass_offset = torch.from_numpy(np.array(self.mass_offset)).float().to(self.device)
        else:
            self.mass_offset = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch.from_numpy(np.array(self.com_displacement)).float().to(self.device)
        else:
            self.com_displacement = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(calf_names)):
            self.calf_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], calf_names[i])
        
        link_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.feet_link_index = torch.tensor([link_dict[_name] for _name in feet_names]).to(dtype=torch.long, device=self.device)

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])
        self.base_contact_indice = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "base")
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        self.allow_initial_contact_indices = torch.zeros(len(allow_initial_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(allow_initial_contact_names)):
            self.allow_initial_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], allow_initial_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_heights_at_points(self, points):
        """ Get vertical projected terrain heights at points 
        points: a tensor of size (num_envs, num_points, 2) in world frame
        """
        points = points.clone()
        num_points = points.shape[1]
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(self.num_envs, num_points, device=self.device, requires_grad=False)
        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def GetFootPositionsInBaseFrame(self):
        """Get the robot's foot position in the base frame."""
        return self.foot_positions_in_base_frame(self.dof_pos)
    
    def GetHipPositionsInBaseFrame(self):
        return torch.tile(self.DEFAULT_HIP_POSITIONS, (self.num_envs, 1, 1))

    def ComputeMotorAnglesFromFootLocalPosition(self, leg_id,
                                                foot_local_position):
        """Use IK to compute the motor angles, given the foot link's local position.

        Args:
        leg_id: The leg index.
        foot_local_position: The foot link's position in the base frame.

        Returns:
        A tuple. The position indices and the angles for all joints along the
        leg. The position indices is consistent with the joint orders as returned
        by GetMotorAngles API.
        """
        motors_per_leg = self.num_dofs // self.feet_indices.shape[0]
        joint_position_idxs = list(
            range(leg_id * motors_per_leg,
                leg_id * motors_per_leg + motors_per_leg))

        joint_angles = self.foot_position_in_hip_frame_to_joint_angle(
            foot_local_position - self.HIP_OFFSETS[leg_id].unsqueeze(dim=0),
            l_hip_sign=(-1)**(leg_id + 1))

        # Return the joing index (the same as when calling GetMotorAngles) as well
        # as the angles.
        return joint_position_idxs, joint_angles
    
    def foot_positions_in_base_frame(self, foot_angles):
        foot_angles = foot_angles.reshape((self.num_envs, 4, 3))
        foot_positions = torch.zeros((self.num_envs, 4, 3), dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(4):
            foot_positions[:, i] = self.foot_position_in_hip_frame(
                foot_angles[:, i], l_hip_sign=(-1)**(i + 1))
        return foot_positions + self.HIP_OFFSETS.unsqueeze(dim=0)
    
    @staticmethod
    def foot_position_in_hip_frame(angles, l_hip_sign=1):
        theta_ab, theta_hip, theta_knee = angles[:, 0], angles[:, 1], angles[:, 2]
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        leg_distance = torch.sqrt(l_up**2 + l_low**2 +
                                  2 * l_up * l_low * torch.cos(theta_knee))
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * torch.sin(eff_swing)
        off_z_hip = -leg_distance * torch.cos(eff_swing)
        off_y_hip = l_hip

        off_x = off_x_hip
        off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
        off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip
        return torch.stack([off_x, off_y, off_z], dim=-1)
    
    @staticmethod
    def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        x, y, z = foot_position[:, 0], foot_position[:, 1], foot_position[:, 2]
        theta_knee = -torch.arccos(
            (x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
            (2 * l_low * l_up))
        l = torch.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * torch.cos(theta_knee))
        theta_hip = torch.arcsin(-x / l) - theta_knee / 2
        c1 = l_hip * y - l * torch.cos(theta_hip + theta_knee / 2) * z
        s1 = l * torch.cos(theta_hip + theta_knee / 2) * y + l_hip * z
        theta_ab = torch.atan2(s1, c1)
        return torch.stack([theta_ab, theta_hip, theta_knee], dim=-1)
    
    def MapContactForceToJointTorques(self, leg_id, contact_force):
        # leg_id: int, force: (num_envs, 3)
        jv = self.ComputeJacobian(leg_id)  # Should be (num_envs, 3, num_dofs)
        # motor_torques_list = np.matmul(contact_force, jv)
        motor_torques_list = contact_force.unsqueeze(dim=1).matmul(jv).squeeze(dim=1)
        motor_torques_dict = {}
        motors_per_leg = self.num_dofs // self.feet_indices.shape[0]
        for torque_id, joint_id in enumerate(
            range(leg_id * motors_per_leg, (leg_id + 1) * motors_per_leg)):
            motor_torques_dict[joint_id] = motor_torques_list[:, torque_id]
        return motor_torques_dict
    
    def ComputeJacobian(self, leg_id):
        return self.jacobian[:, self.feet_link_index[leg_id], :3, 6:]

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        reward = torch.square(self.base_lin_vel[:, 2])
        # is_in_collision = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        base_in_collision = torch.norm(self.contact_forces[:, self.base_contact_indice, :], dim=-1) > 0.1
        reward = reward * (1 - base_in_collision.float())
        return reward
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        reward = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        # is_in_collision = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        base_in_collision = torch.norm(self.contact_forces[:, self.base_contact_indice, :], dim=-1) > 0.1
        reward = reward * (1 - base_in_collision.float())
        return reward
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        # "flat" corresponding to the terrain the robot is on
        '''
        left = [0, 2]
        right = [1, 3]
        front = [0, 1]
        rear = [2, 3]
        terrain_heights_at_foot = self._get_heights_at_points(self.foot_positions[:, :, :2])
        terrain_points = torch.cat([self.foot_positions[:, :, :2], terrain_heights_at_foot.unsqueeze(dim=-1)], dim=-1)
        # issues?
        desired_projecty = (terrain_heights_at_foot[:, right] - terrain_heights_at_foot[:, left]).mean(dim=-1) / torch.clip(
            torch.norm(terrain_points[:, left].mean(dim=1) - terrain_points[:, right].mean(dim=1), dim=-1), min=1e-5)
        desired_projectx = (terrain_heights_at_foot[:, rear] - terrain_heights_at_foot[:, front]).mean(dim=-1) / torch.clip(
            torch.norm(terrain_points[:, rear].mean(dim=1) - terrain_points[:, front].mean(dim=1), dim=-1), min=1e-5)
        desired_projection = torch.stack([desired_projectx, desired_projecty], dim=-1)
        reward = torch.sum(torch.square(self.projected_gravity[:, :2] - desired_projection), dim=1)
        '''
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        reward = torch.sum(torch.square(self.torques), dim=1)
        return reward

    def _reward_dof_vel(self):
        # Penalize dof velocities
        reward = torch.sum(torch.square(self.dof_vel), dim=1)
        return reward
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        reward = torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
        return reward
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        reward = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        return reward
    
    def _reward_action_overshoot(self):
        reward = torch.sum(self.overshoot_buf, dim=-1)
        return reward
    
    def _reward_action_q_diff(self):
        reward = torch.sum(self.q_diff_buf, dim=-1)
        return reward
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        # if (self.contact_forces[0, self.fr_hip_contact_indice, :] > 10).any() \
        #     or (self.contact_forces[0, self.fl_hip_contact_indice, :] > 10).any() \
        #     or (self.contact_forces[0, self.rr_hip_contact_indice, :] > 10).any() \
        #     or (self.contact_forces[0, self.rl_hip_contact_indice, :] > 10).any():
        #     if not hasattr(self, "debug_idx"):
        #         self.debug_idx = 0
        #     print(self.contact_forces[0, self.fr_hip_contact_indice], self.contact_forces[0, self.fl_hip_contact_indice], self.contact_forces[0, self.rr_hip_contact_indice], self.contact_forces[0, self.rl_hip_contact_indice])
        #     img = self.get_camera_image()
        #     image = im.fromarray(img.astype(np.uint8))
        #     filename = "debug%d.png" % self.debug_idx
        #     self.debug_idx += 1
        #     image.save(filename)
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        reward = torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        # is_in_collision = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        base_in_collision = torch.norm(self.contact_forces[:, self.base_contact_indice, :], dim=-1) > 0.1
        reward = reward * (1 - base_in_collision.float())
        return reward
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        reward = torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
        # is_in_collision = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        base_in_collision = torch.norm(self.contact_forces[:, self.base_contact_indice, :], dim=-1) > 0.1
        reward = reward * (1 - base_in_collision.float())
        return reward

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        # is_in_collision = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        base_in_collision = torch.norm(self.contact_forces[:, self.base_contact_indice, :], dim=-1) > 0.1
        rew_airTime = rew_airTime * (1 - base_in_collision.float())
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)