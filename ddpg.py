# individual network settings for each actor + critic pair
# see networkforall for details

from model import Actor, Critic
from torch.optim import Adam
import torch
import numpy as np
import torch.nn.functional as F


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class DDPGAgent:
    def __init__(self, state_size, action_size, config):
        super(DDPGAgent, self).__init__()

        self.actor = Actor(
            state_size,
            action_size,
            config["seed"],
            config["nn_size"],
            config["nn_size"] // 2,
        ).to(device)
        self.critic = Critic(
            state_size * 2,
            action_size * 2,
            config["seed"],
            config["nn_size"],
            config["nn_size"] // 2,
        ).to(device)
        self.actor_target = Actor(
            state_size,
            action_size,
            config["seed"],
            config["nn_size"],
            config["nn_size"] // 2,
        ).to(device)
        self.critic_target = Critic(
            state_size * 2,
            action_size * 2,
            config["seed"],
            config["nn_size"],
            config["nn_size"] // 2,
        ).to(device)

        self.noise = OUNoise(action_size, config)

        # initialize targets same as original networks
        self.soft_update(self.actor_target, self.actor, 1)
        self.soft_update(self.critic_target, self.critic, 1)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=config["lr_actor"])
        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=config["lr_critic"],
            weight_decay=config["weight_decay"],
        )
        self.tau = config["tau"]
        self.gamma = config["gamma"]

    def act(self, states, noise_scale):
        states = torch.from_numpy(states).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(states).cpu().data.numpy()
        self.actor.train()

        action += noise_scale * self.noise.noise()
        action = np.clip(action, -1, 1)
        return action

    def target_act(self, states, noise_scale=0.0):
        states = torch.from_numpy(states).float().to(device)
        action = self.actor_target(states)  # + noise_scale * self.noise.noise()
        return action

    def learn(self, states, actions, rewards, next_states, dones):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states_tensor = torch.cat(states, dim=1).to(device)
        actions_tensor = torch.cat(actions, dim=1).to(device)
        next_states_tensor = torch.cat(next_states, dim=1).to(device)
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next_local = [
            self.actor_target(next_state) for next_state in next_states
        ]
        actions_next_tensor = torch.cat(actions_next_local, dim=1).to(device)

        Q_targets_next = self.critic_target(next_states_tensor, actions_next_tensor)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic(states_tensor, actions_tensor)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = [self.actor(state) for state in states]
        actions_pred_tensor = torch.cat(actions_pred, dim=1).to(device)

        actor_loss = -self.critic(states_tensor, actions_pred_tensor).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )


class OUNoise:
    def __init__(self, action_dimension, config):
        self.action_dimension = action_dimension
        self.mu = config["mu"]
        self.theta = config["theta"]
        self.sigma = config["sigma"]
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
