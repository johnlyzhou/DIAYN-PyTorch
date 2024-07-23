from src.envs.base import Env
from src.envs.base import Step
from gym.spaces import Box
import torch


class PointEnv(Env):
    def __init__(self, max_steps: int = 100):
        super().__init__()
        self.max_steps = max_steps
        self.num_steps = 0
        self._state = None
        self.reset()

    @property
    def observation_space(self):
        return Box(low=-2, high=2, shape=(2,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def reset(self):
        self.num_steps = 0
        self._state = torch.zeros((2,))
        observation = torch.clone(self._state)
        return observation

    def step(self, action):
        self.num_steps += 1
        self._state = self._state + action
        reward = 0
        done = self.num_steps >= self.max_steps
        next_observation = torch.clone(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)
