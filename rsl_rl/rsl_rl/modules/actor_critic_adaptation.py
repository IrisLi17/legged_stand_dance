from rsl_rl.modules.actor_critic import ActorCritic, get_activation
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCriticAdaptation(nn.Module):
    def __init__(
            self, 
            num_obs, 
            num_env_factor,
            num_history_obs, 
            num_actions, 
            num_encoded_dim=32,
            actor_hidden_dims=[256, 256, 256], 
            critic_hidden_dims=[256, 256, 256],
            env_encoder_hidden_dims=[256, 128], 
            adaptation_hidden_dims=[256, 128],
            activation='elu', 
            init_noise_std=1, 
            **kwargs
        ):
        super().__init__()
        activation = get_activation(activation)

        self.num_obs = num_obs
        self.num_env_factors = num_env_factor
        self.num_history_obs = num_history_obs
        self.num_encoded_dim = num_encoded_dim
        
        env_encoder_module = []
        _input_dim = num_env_factor
        for i in range(len(env_encoder_hidden_dims)):
            env_encoder_module.append(nn.Linear(_input_dim, env_encoder_hidden_dims[i]))
            env_encoder_module.append(activation)
            _input_dim = env_encoder_hidden_dims[i]
        env_encoder_module.append(nn.Linear(env_encoder_hidden_dims[-1], num_encoded_dim))
        self.env_encoder_module = nn.Sequential(*env_encoder_module)

        adaptation_module = []
        _input_dim = num_history_obs
        for i in range(len(adaptation_hidden_dims)):
            adaptation_module.append(nn.Linear(_input_dim, adaptation_hidden_dims[i]))
            adaptation_module.append(activation)
            _input_dim = adaptation_hidden_dims[i]
        adaptation_module.append(nn.Linear(adaptation_hidden_dims[-1], num_encoded_dim))
        self.adaptation_module = nn.Sequential(*adaptation_module)
        
        actor_layers = []
        _input_dim = num_obs + num_encoded_dim
        for i in range(len(actor_hidden_dims)):
            actor_layers.append(nn.Linear(_input_dim, actor_hidden_dims[i]))
            actor_layers.append(activation)
            _input_dim = actor_hidden_dims[i]
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = []
        _input_dim = num_obs + num_encoded_dim # TODO: check what should be the input
        for i in range(len(critic_hidden_dims)):
            critic_layers.append(nn.Linear(_input_dim, critic_hidden_dims[i]))
            critic_layers.append(activation)
            _input_dim = critic_hidden_dims[i]
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)
    
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
    
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations, env_factors):
        latent = self.env_encoder_module.forward(env_factors)
        obs = torch.cat([observations, latent], dim=-1)
        mean = self.actor.forward(obs)
        self.distribution = Normal(mean, mean*0. + self.std)
    
    def update_distribution_student(self, observations, history_obs):
        latent = self.adaptation_module.forward(history_obs)
        obs = torch.cat([observations, latent], dim=-1)
        mean = self.actor.forward(obs)
        self.distribution = Normal(mean, mean * 0 + self.std)
    
    def act(self, observations, env_factors, **kwargs):
        self.update_distribution(observations, env_factors)
        return self.distribution.sample()
    
    def act_student(self, observations, history_obs, **kwargs):
        self.update_distribution_student(observations, history_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, ob_dict):
        history_obs = ob_dict["history_obs"]
        obs = ob_dict["obs"]
        # assert flat_obs.shape[1] == self.num_history_obs + self.num_obs
        # history_obs = flat_obs[:, :self.num_history_obs]
        # obs = flat_obs[:, self.num_history_obs:]
        latent = self.adaptation_module.forward(history_obs)
        actions_mean = self.actor.forward(torch.cat([obs, latent], dim=-1))
        return actions_mean
    
    def act_teacher(self, ob_dict):
        env_factor = ob_dict["env_factor"]
        obs = ob_dict["obs"]
        latent = self.env_encoder_module.forward(env_factor)
        actions_mean = self.actor.forward(torch.cat([obs, latent], dim=-1))
        return actions_mean
    
    def evaluate(self, critic_observations, env_factors, **kwargs):
        latent = self.env_encoder_module.forward(env_factors)
        return self.critic.forward(torch.cat([critic_observations, latent], dim=-1))
    
    def create_export_actor(self):
        return ExportActor(self.adaptation_module, self.actor, self.num_history_obs, self.num_obs)
    
    
    # def get_adaptation_loss(self, priv_observations):
    #     observations = torch.narrow(priv_observations, dim=-1, start=0, length=self.num_history_obs)
    #     env_factors = torch.narrow(priv_observations, dim=-1, start=self.num_history_obs, length=self.num_env_factors)
    #     latent = self.adaptation_module.forward(observations)
    #     return torch.mean(0.5 * torch.square(latent - env_factors).sum(dim=-1))

class ExportActor(nn.Module):
    def __init__(self, adaptation_module: nn.Module, actor: nn.Module, 
                 num_history_obs: int, num_obs: int):
        super(ExportActor, self).__init__()
        self.adaptation_module = adaptation_module
        self.actor = actor
        self.num_history_obs = num_history_obs
        self.num_obs = num_obs
    
    def forward(self, x):
        assert x.shape[1] == self.num_history_obs + self.num_obs
        history_obs = x[:, :self.num_history_obs]
        obs = x[:, self.num_history_obs:]
        latent = self.adaptation_module.forward(history_obs)
        actions_mean = self.actor.forward(torch.cat([obs, latent], dim=-1))
        return actions_mean
