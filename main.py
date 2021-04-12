import numpy as np
import torch
import gym
import matplotlib.pyplot as plt

from collections import defaultdict
from agent import Agent
from replay_buffer import UniformReplayBuffer

def DQN(env_name, seed=0, episodes=100, gamma=0.99, q_lr=1e-3,
        steps_per_epoch=3000, start_steps=5000,
        update_after=100, update_every=5, update_target_every=100,
        buffer_size=10000, batch_size=32,
        scheduler=dict(), n_filters=[16, 32],
        double=False, dueling=False, render=False, verbose = 100):
    """
    Implements standard DQN, Dueling DQN and Double DQN
    :param env_name: OpenAI Gym environment name
    :param seed: seed for random generators
    :param episodes: number of episodes
    :param gamma: discounting rate
    :param q_lr: learning rate for the value function
    :param start_steps: amount of steps when action are selected randomly. Helps exploration
    :param update_after: start performing gradient steps after update_after STEPS
    :param update_every: perform gradient steps every update_every STEPS
    :param update_target_every: copy parameters for the target value function every update_every EPISODES
    :param buffer_size: replay buffer size
    :param batch_size: batch size
    :param scheduler_params: parameters for a epsilon scheduler
    :param n_filters: list of filter for the value functions architectures
    :param double: specifies the way the main value function is updated
    :param dueling: specifies usage of dueling architecture for value functions
    :param render: render environment at training stage or not
    :param verbose: print out statistics every verbose episodes
    :return:
    """
    env = gym.make(env_name)  # create an OpenAI Gym environment

    # set all seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # get observation space and action space dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n


    buf = UniformReplayBuffer(buffer_size, obs_dim) # replay buffer
    agent = Agent(obs_dim, act_dim, lr = q_lr, gamma = gamma,
                  double = double, dueling = dueling,
                  n_filters = n_filters, scheduler = scheduler) # agent

    rewards = []
    max_q_values = defaultdict(list)
    
    t = 0  # timestep
    for episode in range(episodes):
        if render:
            env.render()

        done = False
        o, ep_ret = env.reset(), 0
        while not done:

            if t > start_steps:
                # first start_steps steps select actions randomly. Hepls exploration
                a, q = agent.act(torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0), t)
                if q is not None:
                    max_q_values[a].append(q)
            else:
                a = env.action_space.sample()

            o2, r, done, _ = env.step(a)

            ep_ret += r
            buf.store(o, a, r, o2, done)
            o = o2
            
            if t >= update_after:
                # main value function update
                if t % update_every == 0:
                    for _ in range(update_every):
                        batch = buf.sample_batch(batch_size)
                        agent.update_main(batch)

                # target value function update
                if t % update_target_every == 0:
                    agent.update_target()
            t += 1
        

        rewards.append(ep_ret)
        if (episode + 1) % verbose == 0:
            print(f"Episode: {episode + 1:}")
            print(f"\tmean:{np.mean(rewards[-verbose:])} \
            min: {np.min(rewards[-verbose:])} \
            max: {np.max(rewards[-verbose:])}")

    return rewards, max_q_values, agent