import numpy as np
from tqdm import tqdm
import torch

from src import SACAgent
from src.config import get_params
from src.envs.point_env import PointEnv
from src.logger import Logger


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


if __name__ == "__main__":
    params = get_params()

    test_env = PointEnv()
    if isinstance(test_env, PointEnv):
        params['env_name'] = "PointEnv"
        params["n_skills"] = 10

    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.shape[0]
    action_bounds = [torch.from_numpy(test_env.action_space.low),
                     torch.from_numpy(test_env.action_space.high)]

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    print("params:", params)
    del test_env, n_states, n_actions, action_bounds

    env = PointEnv()

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)
    logger = Logger(agent, **params)

    if not params["train_from_scratch"]:
        episode, last_logq_zs, np_rng_state, *env_rng_states, torch_rng_state, random_rng_state = logger.load_weights()
        agent.hard_update_target_network()
        min_episode = episode
        np.random.set_state(np_rng_state)
        # env.np_random.set_state(env_rng_states[0])
        env.observation_space.np_random.set_state(env_rng_states[1])
        env.action_space.np_random.set_state(env_rng_states[2])
        agent.set_rng_states(torch_rng_state, random_rng_state)
        print("Keep training from previous run.")

    else:
        min_episode = 0
        last_logq_zs = 0
        np.random.seed(params["seed"])
        torch.random.manual_seed(params["seed"])
        # env.seed(params["seed"])
        # env.observation_space.seed(params["seed"])
        # env.action_space.seed(params["seed"])
        print("Training from scratch.")

    logger.on()
    for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
        z = np.random.choice(params["n_skills"], p=p_z)
        state = env.reset()
        state = concat_state_latent(state, z, params["n_skills"])
        episode_reward = 0
        logq_zses = []

        max_n_steps = min(params["max_episode_len"], env.max_steps)

        for step in range(1, 1 + max_n_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = concat_state_latent(next_state, z, params["n_skills"])
            agent.store(state, z, done, action, next_state)
            logq_zs = agent.train()
            if logq_zs is None:
                logq_zses.append(last_logq_zs)
            else:
                logq_zses.append(logq_zs)
            episode_reward += reward
            state = next_state
            if done:
                break
        logger.log(episode,
                   episode_reward,
                   z,
                   sum(logq_zses) / len(logq_zses),
                   )
