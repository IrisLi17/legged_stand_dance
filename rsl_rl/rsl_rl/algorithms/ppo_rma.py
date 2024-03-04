from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage.rollout_storage import RMARolloutStorage
from rsl_rl.modules.actor_critic_adaptation import ActorCriticAdaptation
import torch
import torch.nn as nn
import torch.optim as optim


class PPORMA(PPO):
    def __init__(self,
                 actor_critic: ActorCriticAdaptation,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 adaptation_learning_rate=1e-3,
                 phase="teacher", # "distill", "improve"
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.adaptation_optimizer = optim.Adam(self.actor_critic.parameters(), lr=adaptation_learning_rate)
        self.transition = RMARolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        if phase == "teacher":
            self.fit_ppo = True
            self.fit_adaptation = True
            self.ppo_teacher = True
        elif phase == "distill":
            self.fit_ppo = False
            self.fit_adaptation = True
            self.ppo_teacher = None
        elif phase == "improve":
            self.fit_ppo = True
            self.fit_adaptation = False
            self.ppo_teacher = True
            params_dict = [{'params': self.actor_critic.adaptation_module.parameters(), 'lr': self.learning_rate},              
                           {'params': self.actor_critic.actor.parameters(), 'lr': 0.0},
                           {'params': self.actor_critic.critic.parameters(), 'lr': self.learning_rate},
                           {'params': self.actor_critic.env_encoder_module.parameters(), 'lr': 0.0}]
            self.optimizer = torch.optim.Adam(params_dict)
        else:
            raise NotImplementedError
    
    def init_storage(self, num_envs, num_transitions_per_env, obs_shape, env_factor_shape, history_obs_shape, action_shape):
        self.storage = RMARolloutStorage(num_envs, num_transitions_per_env, obs_shape, env_factor_shape, history_obs_shape, action_shape, self.device)
    
    def act(self, obs_dict, use_teacher=True):
        # Compute the actions and values
        obs = obs_dict["obs"]
        env_factors = obs_dict["env_factor"]
        history_obs = obs_dict["history_obs"]
        critic_obs = obs
        if use_teacher:
            self.transition.actions = self.actor_critic.act(obs, env_factors).detach()
        else:
            self.transition.actions = self.actor_critic.act_student(obs, history_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs, env_factors).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        if self.actor_critic.action_mean is not None:
            self.transition.action_mean = self.actor_critic.action_mean.detach()
            self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.env_factors = env_factors
        self.transition.history_observations = history_obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def compute_returns(self, last_critic_obs, env_factors):
        last_values= self.actor_critic.evaluate(last_critic_obs, env_factors).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
    
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_loss = 0
        mean_clipped_ratio = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, env_factors_batch, history_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:


                if self.fit_ppo:
                    if self.ppo_teacher:
                        self.actor_critic.act(obs_batch, env_factors_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                    else:
                        self.actor_critic.act_student(obs_batch, history_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                    actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                    value_batch = self.actor_critic.evaluate(obs_batch, env_factors_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                    mu_batch = self.actor_critic.action_mean
                    sigma_batch = self.actor_critic.action_std
                    entropy_batch = self.actor_critic.entropy

                    # KL
                    if self.desired_kl != None and self.schedule == 'adaptive':
                        with torch.inference_mode():
                            kl = torch.sum(
                                torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                            kl_mean = torch.mean(kl)

                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                            
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = self.learning_rate


                    # Surrogate loss
                    ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                    surrogate = -torch.squeeze(advantages_batch) * ratio
                    surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                    1.0 + self.clip_param)
                    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                    clipped_ratio = torch.sum(surrogate_clipped > surrogate) / surrogate_clipped.shape[0]

                    # Value function loss
                    if self.use_clipped_value_loss:
                        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                        self.clip_param)
                        value_losses = (value_batch - returns_batch).pow(2)
                        value_losses_clipped = (value_clipped - returns_batch).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (returns_batch - value_batch).pow(2).mean()
                
                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                    # Gradient step
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    mean_value_loss += value_loss.item()
                    mean_surrogate_loss += surrogate_loss.item()
                    mean_clipped_ratio += clipped_ratio.item()

                if self.fit_adaptation:
                    stu_env_latent = self.actor_critic.adaptation_module(history_obs_batch)
                    with torch.no_grad():
                        teacher_env_latent = self.actor_critic.env_encoder_module(env_factors_batch)
                    adaptation_loss = nn.functional.mse_loss(stu_env_latent, teacher_env_latent)
                    self.adaptation_optimizer.zero_grad()
                    adaptation_loss.backward()
                    self.adaptation_optimizer.step()
                    mean_adaptation_loss += adaptation_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_clipped_ratio /= num_updates
        mean_adaptation_loss /= num_updates
        self.storage.clear()

        loss_dict = {
            "value_loss": mean_value_loss,
            "surrogate_loss": mean_surrogate_loss,
            "clipped_ratio": mean_clipped_ratio,
            "adaptation_loss": mean_adaptation_loss
        }
        return loss_dict
