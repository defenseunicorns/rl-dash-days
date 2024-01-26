import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

import matplotlib.pyplot as plt
import numpy as np
from networks.cnn import CNN, init_weights
from networks.buffer import PPOBuffer
from algos.logger import Logger
from envs.mspacman import JOYSTICK_TRANSLATION
import random

class PPOPolicy(nn.Module):
    """Actor critic policy for PPO
        :param height: height of obs
        :param width: width of obs
        :param num_frames: number of frames in obs
        :param num_actions: action space to project to (set to 1 for critic)
        :dev: torch device to use
        :param action_std: hyperparameter to develope distribution
    """
    def __init__(self, height, width, num_frames, num_actions, dev, action_std=.5):
        super().__init__()
        self.dev = dev
        self.actor = CNN(height, width, num_frames, num_actions, q_learn=False)
        self.critic = CNN(height, width, num_frames, num_actions=1, q_learn=False)

        self.action_var = torch.full((num_actions, ), action_std*action_std).to(self.dev)

    def act(self, state):
        """Infers an action from a state and updates the buffer
            :param state: observation from environment
            :returns: action sampled
        """
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.dev)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action, action_logprob

    def evaluate(self, states, actions):
        """Evaluation function that is used to gather elements to calculate GAE
            :param states: batch of states shape (bs, num_frames, height, width)
            :param actions: batch of actions shape (bs, num_actions)
        """
        V_s = self.critic(states)                                   #(bs,1)

        action_means = self.actor(states)                           #(bs, num_actions)
        action_vars = self.action_var.expand_as(action_means)
        cov_mat = torch.diag_embed(action_vars).to(self.dev)        #(bs,num_actions,num_actions)
        dist = MultivariateNormal(action_means, cov_mat)
        action_logprobs = dist.log_prob(actions)                    #(bs)
        dist_entropy = dist.entropy()                               #(bs)

        return action_logprobs, torch.squeeze(V_s), dist_entropy

class PPO:
    """
        Proximal policy optimization algorithm

        :param name: for checkpointing / data saving
        :param env: Environment object
        :param reward_fcn: reward function to interact with the environment
        :param gamma: reward discount
        :param num_frames: number of frames in a state
        :param num_actions: ppo action space
        :param eps_clip: clipping for ppo updates
        :param timesteps_per_batch: Max timesteps for a single game
        :param update_timesteps: how many timesteps to update model
        :param k_epochs: number of stochastic model updates
    """
    def __init__(self, name, env, reward_fcn,
                 gamma=.95, num_frames=4,
                 num_actions=2, training=True,
                 eps_clip = .2,
                 timesteps_per_batch=2048,
                 update_timesteps=5000,
                 k_epochs=10, **reward_kwargs):
        self.name = name
        self.path = f'./models/checkpoint_{self.name}.pt'
        self.reward_fcn = reward_fcn
        self.num_frames = num_frames
        self.device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
        self.gamma = gamma
        self.env = env
        self.start = self.reset_env()
        _, height, width = self.start.size()
        self.num_actions = 2
        self.policy = PPOPolicy(height, width, num_frames, num_actions, self.device)
        self.policy.to(self.device).train()
        self.policy.apply(init_weights)

        if training:
            self.memory = PPOBuffer()
            self.target = PPOPolicy(height, width, num_frames, num_actions, dev=self.device)
            self.target.to(self.device).eval()
            
            self.opt = torch.optim.Adam(self.policy.parameters(), lr=.00025, betas=[.9, .990])
            self.policy.train()
            self.timesteps_per_batch = timesteps_per_batch
            self.update_timesteps = update_timesteps
            self.max_episodes = max_episodes
            self.k_epochs = k_epochs
            self.iteration = 0
            self.eps_clip = eps_clip
            self.loss = nn.MSELoss()

        self.reward_kwargs = reward_kwargs

    def reset_env(self):
        """resets the environment"""
        state, score, terminal, lives, frame_number = self.env.reset()
        self.lives = lives
        return state

    def unpack_memory(self):
        rewards = []
        discounted_reward = 0
        for reward, terminal in zip(reversed(self.memory.rewards), reversed(self.memory.terms)):
            if terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        #normalize rewards
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        #supporting tensors for loss calculation
        #detaching because the target calculated all of this
        old_states = torch.squeeze(torch.stack(self.memory.states).to(self.device)).detach()
        old_actions = torch.squeeze(torch.stack(self.memory.actions).to(self.device)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs).to(self.device)).detach()

        return old_states, old_actions, old_logprobs, rewards

    def save_and_update(self):
        torch.save(self.policy.state_dict(), self.path)
        self.target.load_state_dict(self.policy.state_dict())
    
    def update(self, running_stats):
        old_states, old_actions, old_logprobs, rewards = self.unpack_memory()

        for e in range(self.k_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()

            #actor loss
            l1 = ratios * advantages
            l2 = torch.clamp(ratios, 1-self.eps_clip, 1+ self.eps_clip) * advantages
            actor_loss = - torch.min(l1, l2)

            #critic loss
            critic_loss = .5 * self.loss(rewards, state_values) - .01 * dist_entropy

            #total loss
            loss = actor_loss + critic_loss

            self.opt.zero_grad()
            loss.mean().backward()
            self.opt.step()

        running_stats['loss'] = loss.mean().item()
        running_stats['count'] = 1
        self.save_and_update()

        return running_stats

    def train(self, epochs=10000, start_iter=0):
        self.logger = Logger()
        running_stats = self.logger.init_stats()
        for e in range(epochs):
            state, score, terminal, lives, frame = self.env.reset()
            self.lives = lives
            running_stats = self.logger.init_epoch(running_stats)
            for i in range(1, self.timesteps_per_batch):
                state = state.unsqueeze(0)
                self.memory.push_state(state)
                self.iteration += 1
                action, logprob = self.target.act(state.to(self.device))
                state, score, terminal, lives, frames = self.env.step(action.squeeze(0))
                reward = self.reward_fcn(score, lives, self.lives, terminal, **self.reward_kwargs)
                self.lives = lives
                running_stats = self.logger.update_running_stats(running_stats, score)
    
                self.memory.push(action, reward, terminal, logprob)
    
                if self.iteration % self.update_timesteps == 0:
                    running_stats = self.update(running_stats)
                    self.memory.clear()
    
                if terminal:
                    break

            running_stats = self.logger.end_epoch(running_stats)
            
            if e%100 == 0:
                stop = self.logger.update_overall_stats(running_stats, None, e, epochs)
                running_stats = self.logger.init_stats()