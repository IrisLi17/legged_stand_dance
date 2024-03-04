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

import time
import os
from collections import deque
import statistics

import torch
import wandb
import numpy as np

from rsl_rl.algorithms.ppo_rma import PPORMA
from rsl_rl.modules import ActorCriticAdaptation
from rsl_rl.env import VecEnv


class OnPolicyRMARunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.use_wandb = train_cfg["use_wandb"]
        self.device = device
        self.env = env
        if self.alg_cfg["phase"] == "teacher":
            self.cfg["use_teacher_prob"] = 1.0
        elif self.alg_cfg["phase"] == "distill":
            self.cfg["use_teacher_prob"] = 0.8
        elif self.alg_cfg["phase"] == "improve":
            self.cfg["use_teacher_prob"] = 0.0
        else:
            raise NotImplementedError
        actor_critic = ActorCriticAdaptation(self.env.num_obs * self.env.num_stacked_obs,
                                             self.env.num_env_factors,
                                             self.env.num_obs * self.env.num_history,
                                             self.env.num_actions,
                                             **self.policy_cfg).to(self.device)
        self.alg = PPORMA(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs * self.env.num_stacked_obs], 
                              [self.env.num_env_factors], [self.env.num_obs * self.env.num_history], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs_dict = self.env.get_observations()
        obs = obs_dict["obs"]
        env_factor = obs_dict["env_factor"]
        history_obs = obs_dict["history_obs"]
        # privileged_obs = self.env.get_privileged_observations()
        critic_obs = obs
        obs, env_factor, history_obs, critic_obs = obs.to(self.device), env_factor.to(self.device), history_obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            _use_teacher_rollout = True
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if i % self.cfg["sample_rollout_actor_interval"] == 0:
                        _use_teacher_rollout = np.random.uniform(0, 1) < self.cfg["use_teacher_prob"]
                    actions = self.alg.act(obs_dict, use_teacher=_use_teacher_rollout)
                    obs_dict, _, rewards, dones, infos = self.env.step(actions)
                    obs = obs_dict["obs"]
                    env_factor = obs_dict["env_factor"]
                    history_obs = obs_dict["history_obs"]
                    critic_obs = obs
                    obs, env_factor, history_obs = obs.to(self.device), env_factor.to(self.device), history_obs.to(self.device)
                    critic_obs, rewards, dones = critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, env_factor)
            
            loss_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_log_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_log_dict[f'Episode/{key}'] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        entropy = self.alg.actor_critic.entropy.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_log_dict['Loss/value_function'] = locs['loss_dict']['value_loss']
        wandb_log_dict['Loss/surrogate'] = locs['loss_dict']['surrogate_loss']
        wandb_log_dict['Loss/clipped_ratio'] = locs['loss_dict']['clipped_ratio']
        wandb_log_dict['Loss/adaptation'] = locs['loss_dict']['adaptation_loss']
        wandb_log_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_log_dict['Policy/mean_entropy'] = entropy.item()
        wandb_log_dict['Perf/total_fps'] = fps
        wandb_log_dict['Perf/collection time'] = locs['collection_time']
        wandb_log_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_log_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_log_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
        if self.use_wandb:
            wandb.log(wandb_log_dict, step=self.tot_timesteps)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['loss_dict']['value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['loss_dict']['surrogate_loss']:.4f}\n"""
                          f"""{'Clipped ratio:':>{pad}} {locs['loss_dict']['clipped_ratio']:.4f}\n"""
                          f"""{'Adaptation loss:':>{pad}} {locs['loss_dict']['adaptation_loss']:.4f}\n"""
                          f"""{'Mean action entropy:':>{pad}} {entropy.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['loss_dict']['value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['loss_dict']['surrogate_loss']:.4f}\n"""
                          f"""{'Clipped ratio:':>{pad}} {locs['loss_dict']['clipped_ratio']:.4f}\n"""
                          f"""{'Adaptation loss:':>{pad}} {locs['loss_dict']['adaptation_loss']:.4f}\n"""
                          f"""{'Mean action entropy:':>{pad}} {entropy.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'adaptation_optimizer_state_dict': self.alg.adaptation_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.adaptation_optimizer.load_state_dict(loaded_dict['adaptation_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        # return self.alg.actor_critic.act_teacher
        return self.alg.actor_critic.act_inference
