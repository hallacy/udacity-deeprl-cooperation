# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import numpy as np
from collections import namedtuple, deque
import random

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class MADDPG:
    def __init__(self, state_size, action_size, config):
        super(MADDPG, self).__init__()

        # critic input = states*2 + actions*2 = 48+4=52
        self.maddpg_agent = [
            DDPGAgent(state_size, action_size, config),
            DDPGAgent(state_size, action_size, config),
        ]

        self.discount_factor = config["gamma"]
        self.tau = config["tau"]
        self.iter = 0

    def act(self, states_all_agents, noise_scale):
        """get actions from all agents in the MADDPG object"""
        actions = [
            agent.act(np.expand_dims(states, axis=0), noise_scale)
            for agent, states in zip(self.maddpg_agent, states_all_agents)
        ]
        return actions

    def noise_reset(self):
        for agent in self.maddpg_agent:
            agent.noise.reset()

    def learn(self, samples, noise_scale):
        """update the critics and actors of all the agents """
        # Calculate next actions, call each agent and update
        states, actions, rewards, next_states, dones = samples

        for agent in self.maddpg_agent:
            agent.learn(states, actions, rewards, next_states, dones)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, num_agents):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.num_agents = num_agents
        # self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        states = [
            torch.from_numpy(
                np.vstack([e.state[iLoop] for e in experiences if e is not None])
            )
            .float()
            .to(device)
            for iLoop in range(self.num_agents)
        ]
        actions = [
            torch.from_numpy(
                np.vstack([e.action[iLoop] for e in experiences if e is not None])
            )
            .float()
            .to(device)
            for iLoop in range(self.num_agents)
        ]
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = [
            torch.from_numpy(
                np.vstack([e.next_state[iLoop] for e in experiences if e is not None])
            )
            .float()
            .to(device)
            for iLoop in range(self.num_agents)
        ]
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
