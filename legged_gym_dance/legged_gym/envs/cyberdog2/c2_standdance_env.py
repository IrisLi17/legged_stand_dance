import torch
from collections import deque
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.cyberdog2.c2_env import CyberEnv
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse, quat_conjugate
from legged_gym.utils.math import wrap_to_pi, quat_apply_yaw_inverse, quat_apply_yaw
import numpy as np
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import pickle
import os, copy
from PIL import Image as im
from PIL import ImageDraw


class CyberStandDanceEnv(CyberEnv):
    def _reward_lift_up(self):
        # root_height = self.root_states[:, 2]
        # # four leg stand is ~0.28
        # # sit height is ~0.385
        # reward = torch.exp(root_height - self.cfg.rewards.lift_up_threshold) - 1
        root_height = self.root_states[:, 2]
        delta_height = root_height - self.cfg.rewards.liftup_target
        error = torch.square(delta_height)
        reward = torch.exp(- error / self.cfg.rewards.tracking_liftup_sigma) #use tracking sigma
        return reward

    def _reward_lift_up_linear(self):
        root_height = self.root_states[:, 2]
        reward = (root_height - self.cfg.rewards.lift_up_threshold[0]) / (self.cfg.rewards.lift_up_threshold[1] - self.cfg.rewards.lift_up_threshold[0])
        reward = torch.clamp(reward, 0., 1.)
        return reward
    
    def _reward_tracking_hand_pos(self): # new reward for hand pos tracking
        # mask=torch.zeros(self.num_envs,device=self.device)
        # mask[self.episode_length_buf>=50]=1
        error = torch.sum(torch.square(self.target_hand_pos[:,:] - self.hand_positions[:,:]), dim=1)
        reward = torch.exp(- error / self.cfg.rewards.tracking_pos_sigma)#*mask
        root_height = self.root_states[:, 2]
        cond1 = root_height > 0.37
        forward = quat_apply(self.base_quat, self.forward_vec)
        upright_vec = quat_apply_yaw(self.base_quat, self.upright_vec)
        cond2 = (torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)) > 0.2 ##new added!
        cond0 = self.episode_length_buf > self.cfg.rewards.before_handtrack_steps
        # heading = wrap_to_pi(self._get_cur_heading())
        # cond3 = torch.abs(heading) < 0.2
        reward = reward * cond0.float() * cond1.float() * cond2.float() #* cond3.float()
        return reward
    
    def _generate_random_target(self):
        if not self.cfg.env.use_fix_target:
            ret=torch.rand(self.num_envs,6,dtype=torch.float,device=self.device)
            '''FL: 
            x: [-0.1,-0.05] y:[-0.05,0.18] z:[-0.16,0.10] 0.3
            x: [-0.05,0.15] y:[-0.01,0.25] z:[-0.28,0] 0.4
            x: [0.15,0.43] y:[-0.01,0.20] z:[-0.10,0.05] 0.3
            '''
            mask_up=torch.zeros(self.num_envs,device=self.device,dtype=bool)
            mask_up[ret[:,0]>0.7]=1
            mask_down=torch.zeros(self.num_envs,device=self.device,dtype=bool)
            mask_down[ret[:,0]<0.3]=1
            mask_mid=(mask_up|mask_down)
            mask_mid=~mask_mid
            
            ret[:,0]=torch.rand(self.num_envs,dtype=torch.float,device=self.device)
            ret[mask_down,0]=ret[mask_down,0]*(-0.05-(-0.1))-0.1
            ret[mask_mid,0]=ret[mask_mid,0]*(0.15-(-0.05))-0.05
            ret[mask_up,0]=ret[mask_up,0]*(0.43-0.15)+0.15
            
            ret[mask_up,1]=ret[mask_up,1]*(0.2-(-0.01))-0.01
            ret[mask_up,2]=ret[mask_up,2]*(0.05-(-0.1))-0.1
            ret[mask_down,1]=ret[mask_down,1]*(0.18-(-0.05))-0.05
            ret[mask_down,2]=ret[mask_down,2]*(0.10-(-0.16))-0.16
            ret[mask_mid,1]=ret[mask_mid,1]*(0.25-(-0.01))-0.01
            ret[mask_mid,2]=ret[mask_mid,2]*(0-(-0.28))-0.28
            '''FR: 
            x: [-0.1,-0.05] y:[-0.18,0.05] z:[-0.16,0.10] 0.3
            x: [-0.05,0.15] y:[-0.25,0.01] z:[-0.28,0] 0.4
            x: [0.15,0.43] y:[-0.20,0.01] z:[-0.10,0.05] 0.3
            '''
            mask_up=torch.zeros(self.num_envs,device=self.device,dtype=bool)
            mask_up[ret[:,3]> 0.7]=1
            mask_down=torch.zeros(self.num_envs,device=self.device,dtype=bool)
            mask_down[ret[:,3]<0.3]=1
            mask_mid=(mask_up|mask_down)
            mask_mid=~mask_mid
            
            ret[:,3]=torch.rand(self.num_envs,dtype=torch.float,device=self.device)
            ret[mask_down,3]=ret[mask_down,3]*(-0.05-(-0.1))-0.1
            ret[mask_mid,3]=ret[mask_mid,3]*(0.15-(-0.05))-0.05
            ret[mask_up,3]=ret[mask_up,3]*(0.43-0.15)+0.15
            
            ret[mask_up,4]=ret[mask_up,4]*(0.01-(-0.20))-0.20
            ret[mask_up,5]=ret[mask_up,5]*(0.05-(-0.1))-0.1
            ret[mask_down,4]=ret[mask_down,4]*(0.05-(-0.18))-0.18
            ret[mask_down,5]=ret[mask_down,5]*(0.10-(-0.16))-0.16
            ret[mask_mid,4]=ret[mask_mid,4]*(0.01-(-0.25))-0.25
            ret[mask_mid,5]=ret[mask_mid,5]*(0-(-0.28))-0.28
        else:
            ret=torch.zeros(self.num_envs,6,dtype=torch.float,device=self.device)
            mask_=((self.episode_length_buf - 1 - self.cfg.rewards.before_handtrack_steps)//int(self.cfg.env.target_cycle))
            # num=mask_.clamp_(max=self.cfg.env.num_fix_targets-1)
            num = mask_ % self.cfg.env.num_fix_targets
            for i in range(self.cfg.env.num_fix_targets):
                ret[num==i,:]=torch.Tensor(self.cfg.env.fix_target[i]).to(self.device)
        return ret
    
    def _obtain_target_toward_random_point(self):
        '''
        To get current target cmd
        '''
        self.last_big_target[(self.episode_length_buf - 1) == self.cfg.rewards.before_handtrack_steps] = \
            self._obtain_hand_pos_in_base_frame()[(self.episode_length_buf - 1) == self.cfg.rewards.before_handtrack_steps]
        new_goal_mask = (self.episode_length_buf - 1 - self.cfg.rewards.before_handtrack_steps) % int(self.cfg.env.target_cycle) == 0
        switch_mask = torch.logical_and(
            self.episode_length_buf - 1 > self.cfg.rewards.before_handtrack_steps,
            new_goal_mask
        )
        self.last_big_target[switch_mask] = self.next_big_target[switch_mask]
        self.next_big_target[new_goal_mask] = self._generate_random_target()[new_goal_mask]
        t = ((self.episode_length_buf - 1 - self.cfg.rewards.before_handtrack_steps) % int(self.cfg.env.target_cycle) + 1).unsqueeze(dim=-1) / int(self.cfg.env.target_cycle)
        self.target_hand_pos = self.last_big_target * (1 - t) + self.next_big_target * t
        self.target_hand_pos[self.episode_length_buf - 1 < self.cfg.rewards.before_handtrack_steps] = 0.
        return
    
    def _obtain_hand_pos_in_base_frame(self):
        '''
        Get current hand position in base frame
        '''
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply(quat_conjugate(self.base_quat),cur_footsteps_translated[:, i, :])
        footsteps_in_body_frame=torch.reshape(footsteps_in_body_frame,(self.num_envs,12))
        self.hand_positions=footsteps_in_body_frame[:,:6]
        return self.hand_positions
    
    def _compute_common_obs(self):#+6 hand targets
        obs_commands = self.commands[:, :3]
        common_obs_buf = torch.cat((  # self.base_lin_vel * self.obs_scales.lin_vel,
                                    # self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.projected_forward_vec,
                                    obs_commands * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.clock_inputs[:, -2:],
                                    ),dim=-1)
        if self.cfg.env.obs_t: #default is False
            common_obs_buf = torch.cat([
                common_obs_buf, 
                torch.clamp(self.episode_length_buf / self.cfg.rewards.allow_contact_steps, 0., 1.).unsqueeze(dim=-1)
            ], dim=-1)
        self._obtain_target_toward_random_point()
        common_obs_buf=torch.cat([common_obs_buf,self.target_hand_pos],dim=-1)
        return common_obs_buf
    
    def _get_noise_scale_vec(self, cfg):#+6 hand targets
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_state, dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        start_index = 0
        noise_vec[start_index:start_index + 3] = noise_scales.gravity * noise_level
        noise_vec[start_index + 3: start_index + 6] = noise_scales.gravity * noise_level
        noise_vec[start_index + 6: start_index + 9] = 0.
        noise_vec[start_index + 9:start_index + 21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[start_index + 21:start_index + 33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[start_index + 33:start_index + 45] = 0. # previous actions
        noise_vec[start_index + 45: start_index + 47] = 0. # clock input
        start_index = start_index + 47
        noise_vec[start_index: start_index + 6] = 0 # no noise for hand targets!
        start_index += 6
        assert start_index == self.cfg.env.num_single_state
        return noise_vec
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        # only explicitly allow foot contact in these mercy steps
        self.reset_buf = torch.logical_and(
            torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1),
            torch.logical_not(torch.logical_and(
                torch.any(torch.norm(self.contact_forces[:, self.allow_initial_contact_indices, :], dim=-1) > 1., dim=1),
                self.episode_length_buf <= self.cfg.rewards.allow_contact_steps
            ))
        )
        if torch.any(self.reset_buf):
            pass
            # print("reset due to collision", self.episode_length_buf[self.reset_buf])
            # print("reset due to collision", torch.norm(self.contact_forces[self.cam_env_id, self.termination_contact_indices, :], dim=-1), 
            #       torch.norm(self.contact_forces[self.cam_env_id, self.allow_initial_contact_indices, :], dim=-1), self.episode_length_buf[self.cam_env_id])
        position_protect = torch.logical_and(
            self.episode_length_buf > 3, torch.any(torch.logical_or(
            self.dof_pos < self.dof_pos_hard_limits[:, 0] + 5 / 180 * np.pi, 
            self.dof_pos > self.dof_pos_hard_limits[:, 1] - 5 / 180 * np.pi
        ), dim=-1))
        if torch.any(position_protect):
            pass
            # print("reset due to position protect", self.episode_length_buf[position_protect])
            # print("reset due to position protect", self.dof_pos[self.cam_env_id], "limit", self.dof_pos[self.cam_env_id] - self.dof_pos_hard_limits[:, 0], 
            #       self.dof_pos_hard_limits[:, 1] - self.dof_pos[self.cam_env_id])
        stand_air_condition = torch.logical_and(
            torch.logical_and(self.episode_length_buf > 3, self.episode_length_buf <= self.cfg.rewards.allow_contact_steps),
            torch.any(self.foot_positions[:, -2:, 2] > 0.06, dim=-1)
        )
        if torch.any(stand_air_condition):
            pass
            # print("reset due to standair", self.episode_length_buf[stand_air_condition])
        abrupt_change_condition = torch.logical_and(
            torch.logical_and(self.episode_length_buf > 3, self.episode_length_buf <= self.cfg.rewards.allow_contact_steps),
            # torch.logical_and(self.episode_length_buf > 3, self.episode_length_buf <= 100),
            torch.any(torch.abs(self.dof_pos - self.last_dof_pos) > self.cfg.asset.max_dof_change, dim=-1)
        )
        if torch.any(abrupt_change_condition):
            pass
            # print("reset due to abrupt change", self.episode_length_buf[abrupt_change_condition])
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= position_protect
        self.reset_buf |= stand_air_condition
        self.reset_buf |= abrupt_change_condition
    
    def update_command_curriculum(self, env_ids):
        if "tracking_lin_vel" in self.episode_sums and torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            # self.command_ranges["lin_vel_x"][0] = 0. # no backward vel
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)
            # no side vel
            self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.2, 0., self.cfg.commands.max_curriculum)
        if "tracking_ang_vel" in self.episode_sums and torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_ang_vel"]:
            self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 0.2, 0., self.cfg.commands.max_curriculum)
    
    def _reset_targets(self,env_ids): #new added for convenience
        self.last_big_target[env_ids,:]=self.hand_positions[env_ids]
        new_target=self._generate_random_target()
        self.next_big_target[env_ids,:]=new_target[env_ids,:]
        # self.target_hand_pos[env_ids,:]=self.last_big_target[env_ids,:]+(self.next_big_target[env_ids,:]-self.last_big_target[env_ids,:])/self.cfg.env.target_cycle
    
    def _init_buffers(self): #added handpos & targets
        super()._init_buffers()
        self.last_xy = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.last_heading = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.init_feet_positions = torch.zeros((self.num_envs, 4, 3), dtype=torch.float, device=self.device)
        self.nominal_rear_pos = torch.from_numpy(
            np.array([0.3, -2.0, 1.5, -0.3, -2.0, 1.5])
        ).float().to(self.device)
        self.nominal_rear_pos_sit = torch.from_numpy(
            np.array([0.0, -2.3, 2.1, 0.0, -2.3, 2.1])
        ).float().to(self.device)
        
        self._obtain_hand_pos_in_base_frame()
        self.target_hand_pos=torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self.last_big_target=torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self.next_big_target=torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self._reset_targets(torch.arange(self.num_envs, device=self.device))

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        if self.cfg.commands.discretize:
            conti_velx_cmd = self.commands[env_ids, 0:1]
            self.commands[env_ids, 0:1] = torch.sign(conti_velx_cmd) * torch.round(torch.abs(conti_velx_cmd) / 0.1) * 0.1
            # self.commands[env_ids, 3:4] = torch.sign(conti_heading_cmd) * torch.round(torch.abs(conti_heading_cmd) / (np.pi / 12)) * (np.pi / 12)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.last_xy[env_ids] = torch.clone(self.root_states[env_ids, :2])
        heading = self._get_cur_heading()
        self.last_heading[env_ids] = heading[env_ids]
    
    def post_physics_step(self):
        super().post_physics_step()
        self.last_xy[:] = self.root_states[:, :2]
        self.last_heading[:] = self._get_cur_heading()
        self.init_feet_positions[self.episode_length_buf == 1] = self.foot_positions[self.episode_length_buf == 1]

    def _post_physics_step_callback(self):
        self._obtain_hand_pos_in_base_frame() # added! update handpos
        super()._post_physics_step_callback()

    def _reset_dofs_rand(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.init_dof_pos_range[:, 0] + torch_rand_float(0., 1., (len(env_ids), self.num_dof), device=self.device) * (self.init_dof_pos_range[:, 1] - self.init_dof_pos_range[:, 0])
        
        self.dof_vel[env_ids] = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        # TODO: Important! should feed actor id, not env id
        env_ids_int32 = self.num_actors * env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _reset_robot_states(self, env_ids):
        self._reset_dofs_rand(env_ids)    
        self._reset_root_states_rand(env_ids)
    
    def _reset_root_states_rand(self, env_ids): #changed!
        """ Resets ROOT states position and velocities of selected environmments
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
            #print("in hhere!")
            rand_rpy = torch_rand_float(-np.pi*15/180.0, np.pi*15/180.0, (len(env_ids), 3), device=self.device) #参数：rand在正负15度
            #print("before rand_rpy=",rand_rpy)
            rand_rpy=rand_rpy+torch.Tensor(get_euler_xyz(self.base_init_state[3:7].unsqueeze(0))).to(self.device)
            #print("rand_rpy=",rand_rpy)
            self.root_states[env_ids, 3: 7] = quat_from_euler_xyz(rand_rpy[:, 0], rand_rpy[:, 1], rand_rpy[:, 2])  #!!!changed to +=
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.1, 0.1, (len(env_ids), 6), device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _recompute_ang_vel(self):
        heading = self._get_cur_heading()
        self.commands[:, 2] = torch.clip(
            0.5*wrap_to_pi(self.commands[:, 3] - heading), -self.cfg.commands.clip_ang_vel, self.cfg.commands.clip_ang_vel
        ) * (0.5 * np.pi / self.cfg.commands.clip_ang_vel)
    
    def _reward_nominal_state(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        condition = torch.sum(forward * self.upright_vec, dim=-1) / torch.norm(self.upright_vec, dim=-1) > 0.9
        dof_err = torch.square(self.dof_pos[:, -6:] - self.nominal_rear_pos).mean(dim=-1)
        reward = torch.exp(-dof_err / self.cfg.rewards.dof_sigma) * condition.float()
        return reward
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # actual_lin_vel = torch.cat([-self.base_lin_vel[:, 2:3], self.base_lin_vel[:, 1:2]], dim=-1)
        if not self.cfg.env.vel_cmd:
            self.commands[:, :2] = 0.
        actual_lin_vel = quat_apply_yaw_inverse(self.base_quat, self.root_states[:, 7:10])
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - actual_lin_vel[:, :2]), dim=1)
        reward = torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        forward = quat_apply(self.base_quat, self.forward_vec)
        upright_vec = quat_apply_yaw(self.base_quat, self.upright_vec)
        is_stand = (torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)) > 0.9
        scale_factor_low = self.cfg.rewards.scale_factor_low
        scale_factor_high = self.cfg.rewards.scale_factor_high
        scaling_factor = (torch.clip(
            self.root_states[:, 2], min=scale_factor_low, max=scale_factor_high
        ) - scale_factor_low) / (scale_factor_high - scale_factor_low)
        reward = reward * is_stand.float() * scaling_factor
        return reward
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (around world z axis) or heading commands 
        # TODO: heading computation requires more investigation
        # heading_vec = quat_apply(self.base_quat, self.gravity_vec)
        # heading = torch.atan2(heading_vec[:, 1], heading_vec[:, 0])
        if not self.cfg.env.vel_cmd:
            self.commands[:, 3] = 0.
        heading = self._get_cur_heading()
        if self.cfg.rewards.ang_rew_mode == "heading":
            # old
            heading_error = torch.square(wrap_to_pi(self.commands[:, 3] - heading) / np.pi)
            # new head error
            # heading_error = torch.square(torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1, 1))
            reward = torch.exp(-heading_error / self.cfg.rewards.tracking_sigma)
        elif self.cfg.rewards.ang_rew_mode == "heading_with_pen":
            heading_error = torch.square(wrap_to_pi(self.commands[:, 3] - heading) / np.pi)
            reward = torch.exp(-heading_error / self.cfg.rewards.tracking_sigma)
            est_ang_vel = wrap_to_pi(heading - self.last_heading) / 0.02
            penalty = (torch.abs(est_ang_vel) - 1.0).clamp(min=0)
            reward = reward - 0.1 * penalty
        else:
            # new, trying
            est_ang_vel = wrap_to_pi(heading - self.last_heading) / 0.02
            # ang_vel_error = torch.abs(self.commands[:, 2] - est_ang_vel) / torch.abs(self.commands[:, 2]).clamp(min=1e-6)
            ang_vel_error = torch.abs(self.commands[:, 2] - est_ang_vel)
            reward = torch.exp(-ang_vel_error/self.cfg.rewards.tracking_ang_sigma)
        forward = quat_apply(self.base_quat, self.forward_vec)
        upright_vec = quat_apply_yaw(self.base_quat, self.upright_vec)
        is_stand = (torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)) > 0.9
        # is_stand = quat_apply(self.base_quat, self.forward_vec)[:, 2] > 0.9
        scale_factor_low = self.cfg.rewards.scale_factor_low
        scale_factor_high = self.cfg.rewards.scale_factor_high
        scaling_factor = (torch.clip(
            self.root_states[:, 2], min=scale_factor_low, max=scale_factor_high
        ) - scale_factor_low) / (scale_factor_high - scale_factor_low)
        reward = reward * is_stand.float() * scaling_factor
        return reward
    
    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices[:, -2:] * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.foot_positions[:, -2:, 2]).view(self.num_envs, -1)# - reference_heights
        terrain_at_foot_height = self._get_heights_at_points(self.foot_positions[:, -2:, :2])
        target_height = self.cfg.rewards.foot_target * phases + terrain_at_foot_height + 0.02 
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states[:, -2:])
        condition = self.episode_length_buf > self.cfg.rewards.allow_contact_steps
        rew_foot_clearance = rew_foot_clearance * condition.unsqueeze(dim=-1).float()
        return torch.sum(rew_foot_clearance, dim=1)

    def _reward_rear_air(self):
        contact = self.contact_forces[:, self.feet_indices[-2:], 2] < 1.
        # init_condition = self.root_states[:, 2] < 0.3
        # init_condition = self.episode_length_buf < self.cfg.rewards.allow_contact_steps
        # reward = (torch.all(contact, dim=1) * (~init_condition) + torch.any(contact, dim=1) * init_condition).float()
        calf_contact = self.contact_forces[:, self.calf_indices[-2:], 2] < 1.
        unhealthy_condition = torch.logical_and(~calf_contact, contact)
        reward = torch.all(contact, dim=1).float() + unhealthy_condition.sum(dim=-1).float()
        return reward
    
    def _reward_stand_air(self):
        stand_air_condition = torch.logical_and(
            torch.logical_and(
                self.episode_length_buf < self.cfg.rewards.allow_contact_steps,
                quat_apply(self.base_quat, self.forward_vec)[:, 2] < 0.9
            ), torch.any(self.foot_positions[:, -2:, 2] > 0.03, dim=1)
        )
        return stand_air_condition.float()
    
    def _reward_foot_twist(self):
        vxy = torch.norm(self.foot_velocities[:, :, :2], dim=-1)
        vang = torch.norm(self.foot_velocities_ang, dim=-1)
        condition = self.foot_positions[:, :, 2] < 0.025
        reward = torch.mean((vxy + 0.1 * vang) * condition.float(), dim=1)
        return reward
    
    def _reward_feet_slip(self):
        condition = self.foot_positions[:, :, 2] < 0.03
        # xy lin vel
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        # yaw ang vel
        foot_ang_velocities = torch.square(torch.norm(self.foot_velocities_ang[:, :, 2:] / np.pi, dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(condition.float() * (foot_velocities + foot_ang_velocities), dim=1)
        return rew_slip

    def _reward_foot_shift(self):
        desired_foot_positions = torch.clone(self.init_feet_positions[:, 2:])
        desired_foot_positions[:, :, 2] = 0.02
        rear_foot_shift = torch.norm(self.foot_positions[:, 2:] - desired_foot_positions, dim=-1).mean(dim=1)
        init_ffoot_positions = torch.clone(self.init_feet_positions[:, :2])
        front_foot_shift = torch.norm( torch.stack([
            (init_ffoot_positions[:, :, 0] - self.foot_positions[:, :2, 0]).clamp(min=0), 
            torch.abs(init_ffoot_positions[:, :, 1] - self.foot_positions[:, :2, 1])
        ], dim=-1), dim=-1).mean(dim=1)
        condition = self.episode_length_buf < self.cfg.rewards.allow_contact_steps
        reward = (front_foot_shift + rear_foot_shift) * condition.float()
        return reward
    
    def _reward_front_contact_force(self):
        force = torch.norm(self.contact_forces[:, self.termination_contact_indices[5: 7]], dim=-1).mean(dim=1)
        reward = force
        return reward
    
    def _reward_hip_still(self):
        movement = torch.abs(self.dof_pos.view(self.num_envs, 4, 3)[:, :, 0] - 0.).mean(dim=1)
        condition = self.episode_length_buf < self.cfg.rewards.allow_contact_steps
        reward = movement * condition.float()
        return reward
