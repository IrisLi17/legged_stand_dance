import torch
import torch.nn as nn
import numpy as np
from .actor_critic import ActorCritic

class ActorCriticCNN(ActorCritic):
    def __init__(self, 
                 num_actor_obs, 
                 num_critic_obs, 
                 num_actions, 
                 actor_hidden_dims=[256, 256, 256], 
                 critic_hidden_dims=[256, 256, 256], 
                 activation='elu', 
                 init_noise_std=1, 
                 image_shape=[1, 64, 64],
                 actor_state_encoder=[128],
                 critic_state_encoder=[128],
                 **kwargs):
        # num_actor_obs will pass in num_obs from env, we need to convert to true input dimension to actor
        _proprio_dim = num_actor_obs - int(np.prod(image_shape))
        image_encoder = nn.Sequential(
            nn.Conv2d(image_shape[0], 2 * image_shape[0], 8, 4, 0),
            nn.ReLU(),
            nn.Conv2d(2 * image_shape[0], 4 * image_shape[0], 4, 2, 0),
            nn.ReLU(),
            nn.Conv2d(4 * image_shape[0], 4 * image_shape[0], 3, 1, 0),
            nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            image_feature_dim = image_encoder(torch.ones(1, *image_shape, dtype=torch.float)).shape[-1]
        print("image feature dim", image_feature_dim)
        num_actor_obs = image_feature_dim + actor_state_encoder[-1]
        num_critic_obs = image_feature_dim + critic_state_encoder[-1]

        super().__init__(num_actor_obs, num_critic_obs, num_actions, actor_hidden_dims, critic_hidden_dims, activation, init_noise_std, **kwargs)
        self.state_dim = _proprio_dim
        self.image_shape = image_shape
        self.image_encoder = image_encoder
        actor_state_layer = []
        actor_state_layer.append(nn.Linear(_proprio_dim, actor_state_encoder[0]))
        actor_state_layer.append(nn.ReLU())
        for i in range(len(actor_state_encoder) - 1):
            if i == len(actor_state_encoder) - 2:
                actor_state_layer.append(nn.Linear(actor_state_encoder[i], actor_state_encoder[i + 1]))
            else:
                actor_state_layer.append(nn.Linear(actor_state_encoder[i], actor_state_encoder[i + 1]))
                actor_state_layer.append(nn.ReLU())
        self.actor_state_layers = nn.Sequential(*actor_state_layer)
        critic_state_layer = []
        critic_state_layer.append(nn.Linear(_proprio_dim, critic_state_encoder[0]))
        critic_state_layer.append(nn.ReLU())
        for i in range(len(critic_state_encoder) - 1):
            if i == len(critic_state_encoder) - 2:
                critic_state_layer.append(nn.Linear(critic_state_encoder[i], critic_state_encoder[i + 1]))
            else:
                critic_state_layer.append(nn.Linear(critic_state_encoder[i], critic_state_encoder[i + 1]))
                critic_state_layer.append(nn.ReLU())
        self.critic_state_layers = nn.Sequential(*critic_state_layer)

    def act(self, observations, **kwargs):
        state_obs = torch.narrow(observations, dim=1, start=0, length=self.state_dim)
        image_obs = torch.narrow(observations, dim=1, start=self.state_dim, length=int(np.prod(self.image_shape)))
        image_features = self.image_encoder(image_obs.view((image_obs.shape[0], *self.image_shape)))
        actor_observation = torch.cat([self.actor_state_layers(state_obs), image_features], dim=-1)
        return super().act(actor_observation, **kwargs)
    
    def act_inference(self, observations):
        state_obs = torch.narrow(observations, dim=1, start=0, length=self.state_dim)
        image_obs = torch.narrow(observations, dim=1, start=self.state_dim, length=int(np.prod(self.image_shape)))
        image_features = self.image_encoder(image_obs.view((image_obs.shape[0], *self.image_shape)))
        actor_observation = torch.cat([self.actor_state_layers(state_obs), image_features], dim=-1)
        return super().act_inference(actor_observation)
    
    def evaluate(self, critic_observations, **kwargs):
        state_obs = torch.narrow(critic_observations, dim=1, start=0, length=self.state_dim)
        image_obs = torch.narrow(critic_observations, dim=1, start=self.state_dim, length=int(np.prod(self.image_shape)))
        image_features = self.image_encoder(image_obs.view((image_obs.shape[0], *self.image_shape)))
        critic_observation = torch.cat([self.critic_state_layers(state_obs), image_features], dim=-1)
        return super().evaluate(critic_observation, **kwargs)
