import torch
import torch.nn as nn
from torch.distributions import Categorical
from rsl_rl.modules.actor_critic import get_activation


class MultiheadMLP(nn.Module):
    def __init__(
            self,
            num_input,
            shared_hidden_dims=[256, 256, 256],
            head_dims=[[], [], []],
            num_bins=21,
            activation='elu',
    ):
        super(MultiheadMLP, self).__init__()
        activation_fn = get_activation(activation)
        shared_layers = []
        input_dim = num_input
        for i in range(len(shared_hidden_dims)):
            shared_layers.append(nn.Linear(input_dim, shared_hidden_dims[i]))
            shared_layers.append(activation_fn)
            input_dim = shared_hidden_dims[i]
        self.shared_layers = nn.Sequential(*shared_layers)
        head_layers = []
        for i in range(len(head_dims)):
            per_head_layers = []
            input_dim = shared_hidden_dims[-1]
            for j in range(len(head_dims[i])):
                per_head_layers.append(nn.Linear(input_dim, head_dims[i][j]))
                per_head_layers.append(activation_fn)
                input_dim = head_dims[i][j]
            per_head_layers.append(nn.Linear(input_dim, num_bins))
            per_head_layers = nn.Sequential(*per_head_layers)
            head_layers.append(per_head_layers)
        self.head_layers = nn.ModuleList(head_layers)

    def forward(self, x):
        shared_encoding = self.shared_layers.forward(x)
        head_results = []
        for i, layer in enumerate(self.head_layers):
            head_results.append(layer.forward(shared_encoding))
        return torch.stack(head_results, dim=1) # batch_size, num_head, num_bins


class ActorCriticDiscrete(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        num_bins=21,
                        **kwargs):
        if kwargs:
            print("ActorCriticDiscrete.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticDiscrete, self).__init__()
        
        activation_fn = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        self.num_bins = num_bins

        # Policy
        self.actor = MultiheadMLP(
            mlp_input_dim_a, actor_hidden_dims, [[] for i in range(num_actions)], num_bins, activation
        )

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

        print("Actor", self.actor)
        print("Critic", self.critic)

        self.distribution = None

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return None

    @property
    def action_std(self):
        return None
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        logits = self.actor(observations)
        self.distribution = Categorical(logits=logits)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        actions = self.distribution.sample() # batch_size, num_head
        # scaled to [-1, 1]
        actions = actions.float() / (self.num_bins - 1) * 2 - 1
        return actions
    
    def get_actions_log_prob(self, actions):
        scaled_actions = ((actions + 1) / 2 * (self.num_bins - 1)).long()
        return self.distribution.log_prob(scaled_actions).sum(dim=-1) # batch_size

    def act_inference(self, observations):
        # actions = self.actor(observations)
        actions = self.act(observations)
        return actions

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value