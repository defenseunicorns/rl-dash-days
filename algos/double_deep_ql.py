import gym
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from networks.cnn import CNN, init_weights
from networks.buffer import ReplayBuffer
from algos.logger import Logger
import random

class DDQ:
    """
        Double Deep Q-learning implementation

        :param name: name for checkpointing / data
        :param env: Environment object
        :param reward_fcn: reward function
        :param policy: policy network
        :param target: target network
        :param gamma: reward discount
        :param batch_size: replay buffer batch sizes
        :param env: gym environment
        :param num_frames: number of frames per state (and number per action)
        :param training: boolean -> whether framework is for training or evaluating
    """
    def __init__(self, name, env, reward_fcn,
                 gamma=.95, batch_size=64, num_frames=4,
                 training=True, **reward_kwargs):
        self.name = name
        self.path = f'./models/checkpoint_{self.name}.pt'
        self.reward_fcn = reward_fcn
        self.num_frames = num_frames
        self.device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
        self.start = self.reset_env()
        _, height, width = self.start.size()
        self.num_actions = self.env.action_space
        self.policy = CNN(height, width, num_frames, self.env.action_space, q_learn=True)
        self.policy.to(self.device).train()
        self.policy.apply(init_weights)

        if training:
            self.memory = ReplayBuffer(50000)
            self.target = CNN(height, width, num_frames, self.env.action_space, q_learn=True)
            self.target.apply(init_weights)
            self.loss = nn.SmoothL1Loss()
            self.opt = torch.optim.RMSprop(self.policy.parameters(), lr=.00025, alpha=.95, eps=.01)
            self.target = target.to(self.device).eval()
            self.policy.eval()
        else:
            self.load_eval()

        self.reward_kwargs = reward_kwargs

    def load_tensors(self, sample):
        """See prepare_update for descriptions"""
        dev = self.device
        states, actions, next_states, rewards, non_terms = list(zip(*sample))
        states = torch.cat(states)
        actions = torch.tensor(actions).long().to(dev)
        next_states = torch.cat(next_states)
        rewards = torch.tensor(rewards).long().to(dev)
        non_terms = torch.tensor(non_terms).to(dev)
        next_mask = torch.ones((actions.size(0), self.num_actions))
        curr_mask = F.one_hot(actions, self.num_actions)
        return states, actions, next_states, rewards, non_terms, curr_mask, next_mask

    def prepare_update(self, states, actions, next_states, rewards, non_terms, curr_mask, next_mask):
        """Uses target and policy network to prepare double deep-q update
            :param states: initial states for state transition
            :param next_states: end states for state transition
            :param rewards: reward gained from state transition
            :param non_terms: boolean for whether a transition ended a game
            :curr_mask: one-hot version of actions
            :next_mask: ONES vector for the next action mask
            :returns: end state q-values (chosen by policy and calculated by target)
                and expected_Q_values (calculated by policy with actual action taken)
        """
        dev = self.device
        policy = self.policy.train()
        target = self.target.eval()

        next_Q_vals = policy(next_states.to(dev), next_mask.to(dev))
        next_actions = next_Q_vals.max(1)[1].to(dev)
        next_mask = F.one_hot(next_actions, self.num_actions).to(dev)
        next_Q_vals = target(next_states.to(dev), next_mask)
        next_Q_vals = next_Q_vals.gather(-1, next_actions.unsqueeze(1)).squeeze(-1) * non_terms
        next_Q_vals = (next_Q_vals * self.gamma) + rewards.to(dev)

        expected_Q_vals = policy(states.to(dev), curr_mask.to(dev))
        expected_Q_vals = expected_Q_vals.gather(-1, actions.unsqueeze(1)).squeeze(-1)

        return next_Q_vals, expected_Q_vals

    def fit_buffer(self, sample):
        """Takes a sample of frames from the buffer and does training"""
        policy = self.policy
        target = self.target
        dev = self.device

        #load tensors
        states, actions, next_states, rewards, non_terms, curr_mask, next_mask = self.load_tensors(sample)
        
        #update rule: Q(s,a) -> gamma* max_{a'}Q(s',a') + r
        next_Q_vals, expected_Q_vals = self.prepare_update(states, actions, next_states, rewards,
                                                           non_terms, curr_mask, next_mask)

        self.opt.zero_grad()
        loss = self.loss(expected_Q_vals, next_Q_vals)
        loss.backward()
        for param in policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt.step()

        return loss.item()

    def reset_env(self):
        """resets the environment"""
        state, score, terminal, lives, frame_number = self.env.reset()
        self.lives = lives
        return state

    def get_epsilon_for_iteration(self, iteration):
        """simple linear epsilon function for"""
        return max(.01, 1-(iteration*.99/300000))

    def infer_action(self, state):
        """Uses the policy model to choose best action for state"""
        dev = self.device
        state = state.unsqueeze(0).to(dev)
        mask = torch.ones(1,self.num_actions).to(dev).squeeze(0)
        self.policy.eval()
        q_vals = self.policy(state, mask).squeeze(0)
        return int(q_vals.max(0)[1])

    def q_iteration(self, state, iteration):
        """Processes a single state transition / action choice"""
        env = self.env
        epsilon = self.get_epsilon_for_iteration(iteration)

        # Choose the action
        if random.random() < epsilon:
            action = self.env.sample()
        else:
            action = self.infer_action(state)
        new_state, score, terminal, lives, frame_number = self.env.step(action)
        reward = self.reward_fcn(score, lives, self.lives, terminal, **self.reward_kwargs)
        self.lives = lives

        #handle non-terms for masking later
        non_term = int(not terminal)
        if new_state.shape[0] < 4:
            print(new_state, new_state.shape[0], score, terminal, lives, frame_number)

        #push to buffer
        mem = (state.unsqueeze(0), action,
               new_state.unsqueeze(0), reward, non_term)
        self.memory.push(mem)

        loss = None
        # Sample and fit
        if iteration > 64:
            batch = self.memory.sample(self.batch_size)
            loss= self.fit_buffer(batch)

        return new_state, reward, score, terminal, loss

    def train(self, epochs=10000, start_iter=0, updates=500):
        """Main training loop, saves statistics to class during training
            :param epochs: number of games to play
            :param start_iter: starting iteration (default 0)
            :param udpates: number of epochs before updating target model
        """
        self.logger = Logger()
        self.updates = updates    #when to save models / swap policy & target
        iteration = start_iter
        running_stats = self.logger.init_stats()
        for e in range(epochs):
            terminal = False
            state, reward, terminal, lives, frames = self.env.reset()
            running_stats = self.logger.init_epoch(running_stats)
            while not terminal:
                new_state, reward, score, terminal, loss = self.q_iteration(state, 
                                                            iteration)
                iteration += 1
                running_stats = self.logger.update_running_stats(running_stats, score, loss)
            running_stats = self.logger.end_epoch(running_stats)
            if e%100 == 0:
                eps = self.get_epsilon_for_iteration(iteration)
                stop = self.logger.update_overall_stats(running_stats, eps, e, epochs)
                running_stats = self.logger.init_stats()
            if e%updates == 0:
                torch.save(self.policy.state_dict(), self.path)
                self.load_target(self.path)

    def load_target(self, path):
        self.target.load_state_dict(torch.load(path))
        self.target.eval()

    def load_eval(self):
        """Used before play() method to load policy network"""
        self.policy.load_state_dict(torch.load(self.path))
        self.policy.eval()

    def eval(self):
        state, reward, terminal, lives, frames = self.env.reset()
        im = plt.imshow(self.env.render())
        plt.ion()
        while not terminal:
            action = self.infer_action(state)
            new_state, reward, terminal, lives, frames = self.env.step(action, render=True, im=im)
            state = new_state
        plt.show()
