import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

from src.networks.mlp import MLP
from src.policies.base import Policy


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


class GaussianNetwork(nn.Module):
    def __init__(self, n_states, n_actions, action_bounds, n_hidden_filters=256):
        super(GaussianNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.mu = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()

        self.log_std = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.log_std, initializer="xavier uniform")
        self.log_std.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = Normal(mu, std)
        return dist


class GaussianPolicy(Policy, nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, action_bounds: tuple = None):
        super(GaussianPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # self.network = GaussianNetwork(obs_dim, action_dim, action_bounds, hidden_sizes)
        # self.network = MLP(obs_dim, (hidden_sizes,), hidden_sizes, output_activation=nn.ReLU())
        # self.mu = nn.Linear(in_features=hidden_sizes, out_features=action_dim)
        # self.log_std = nn.Linear(in_features=hidden_sizes, out_features=action_dim)
        self.network = MLP(obs_dim, (hidden_sizes, hidden_sizes), 2 * action_dim)
        self.action_bounds = action_bounds

    def forward(self, observation):
        output = self.network(observation)
        mean, log_std = output[..., :self.action_dim], output[..., self.action_dim:]
        std = log_std.clamp(min=-20, max=2).exp()
        return Normal(mean, std)

    # def forward(self, observation):
    #     x = self.network(observation)
    #     mu = self.mu(x)
    #     log_std = self.log_std(x)
    #     std = log_std.clamp(min=-20, max=2).exp()
    #     return Normal(mu, std)

    def get_action(self, observation):
        dist = self(observation)
        return dist.sample()

    def get_log_prob(self, observation, action):
        dist = self(observation)
        return dist.log_prob(action)

    def sample_or_likelihood(self, observation):
        dist = self(observation)
        pi_action = dist.rsample()
        logp_pi = dist.log_prob(pi_action).sum(dim=-1)
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(dim=1)
        pi_action = F.tanh(pi_action)
        pi_action = self.action_bounds[0] + (pi_action + 1.) * 0.5 * (self.action_bounds[1] - self.action_bounds[0])
        return pi_action, logp_pi.unsqueeze(-1)
