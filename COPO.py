import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pid_lagrange import PIDLagrangian

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, 128)
        self.l5 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))

        residual = a
        a = F.relu(self.l2(a))

        a = F.relu(self.l3(a))

        a = F.relu(self.l4(a))
        a = torch.add(a, residual)

        return self.max_action * torch.tanh(self.l5(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class COPO(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.01,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            alpha=2.5,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.critic_cost = Critic(state_dim, action_dim).to(device)
        self.critic_cost_target = copy.deepcopy(self.critic)
        self.critic_cost_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0

        self._lagrange = PIDLagrangian(
            pid_kp=2.5,
            pid_ki=0.25,
            pid_kd=0.25,
            pid_d_delay=10,
            pid_delta_p_ema_alpha=0.95,
            pid_delta_d_ema_alpha=0.95,
            sum_norm=True,
            diff_norm=False,
            penalty_max=100,
            lagrangian_multiplier_init=1.,
            cost_limit=0., )

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, reward, next_state, not_done, cost = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state)
                           + noise
                           ).clamp(-self.max_action,
                                   self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

            cost_target_Q1, cost_target_Q2 = self.critic_cost_target(next_state, next_action)
            cost_target_Q = torch.min(cost_target_Q1, cost_target_Q2)
            cost_target_Q = cost + not_done * self.discount * cost_target_Q

        current_Q1, current_Q2 = self.critic(state, action)

        current_cost_Q1, current_cost_Q2 = self.critic_cost(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        cirtic_cost_loss = F.mse_loss(current_cost_Q1, cost_target_Q) + F.mse_loss(current_cost_Q2, cost_target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic_cost_optimizer.zero_grad()
        cirtic_cost_loss.backward()
        self.critic_cost_optimizer.step()

        if self.total_it % self.policy_freq == 0:

            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            cost_Q = self.critic_cost.Q1(state, pi)
            lmbda = self.alpha / (Q - self._lagrange.lagrangian_multiplier * cost_Q).abs().mean().detach()

            actor_loss = -lmbda * (Q - self._lagrange.lagrangian_multiplier * cost_Q).mean() + F.mse_loss(pi, action)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            Jc = cost_Q
            Jc = Jc.detach()
            self._lagrange.pid_update(torch.mean(Jc))

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_cost.parameters(), self.critic_cost_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        torch.save(self.actor, "Offline_actornet_COPO_PID_Lagrangian.pth")
        return critic_loss.item()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
