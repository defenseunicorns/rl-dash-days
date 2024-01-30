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

class DSQ:
    """
        Novel implementation of Lin et. al's split Q-learning for Deep RL

        :param name: name for checkpointing / data
        :param env: Environment object
        :param reward_fcn: reward function
        :param gamma: reward discount
        :param batch_size: replay buffer batch sizes
        :param num_frames: number of frames per state (and number per action)
        :param reward_weight: instantaneous reward scaling [.1,100] typically
        :param punishment_weight: instantaneous punishment scaling [.1,100]
        :param lambda_r: gradient scaling for rewards [.1,1] typically
        :param lambda_p: gradient scaling for punishments [.1,1]
        :param training: If False cuts down on memory usage
    """
    def __init__(self, name, env, reward_fcn,
                 gamma=.95, batch_size=64, num_frames=4,
                 reward_weight=1, punishment_weight=1,
                 lambda_r=1, lambda_p=1, training=True, **reward_kwargs):
        self.name = name
        self.r_path = f'./models/checkpoint_{self.name}_reward.pt'
        self.p_path = f'./models/checkpoint_{self.name}_punish.pt'
        self.reward_fcn = reward_fcn
        self.num_frames = num_frames
        self.device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
        self.gamma = gamma
        self.batch_size = batch_size
        self.w_r = reward_weight
        self.lambda_r = lambda_r
        self.w_p = punishment_weight
        self.lambda_p = lambda_p
        self.env = env
        self.start = self.reset_env()
        _, height, width = self.start.size()
        self.num_actions = self.env.action_space
        self.render = lambda : plt.imshow(env.render())
        self.reward_pol = CNN(height, width,
                             num_frames, self.env.action_space, q_learn=True)
        self.punish_pol = CNN(height, width,
                             num_frames, self.env.action_space, q_learn=True)
        self.reward_pol.to(self.device).train()
        self.punish_pol.to(self.device).train()
        self.loss = nn.SmoothL1Loss()
        self.reward_pol.apply(init_weights)
        self.punish_pol.apply(init_weights)

        if training:
            self.memory = ReplayBuffer(50000)
            self.reward_tar = CNN(height, width,
                                     num_frames, self.env.action_space, q_learn=True).to(self.device).eval()
            self.punish_tar = CNN(height, width,
                                    num_frames, self.env.action_space, q_learn=True).to(self.device).eval()
            self.reward_tar.apply(init_weights)
            self.punish_tar.apply(init_weights)
            self.loss = nn.SmoothL1Loss()
            self.opt_r = torch.optim.RMSprop(self.reward_pol.parameters(), lr=.00025, alpha=.95, eps=.01)
            self.opt_p = torch.optim.RMSprop(self.punish_pol.parameters(), lr=.00025, alpha=.95, eps=.01)
        else:
            self.load_eval()
        

        self.reward_kwargs = reward_kwargs

    def reset_env(self):
        """resets the environment"""
        state, score, terminal, lives, frame_number = self.env.reset()
        self.lives = lives
        return state

    def load_tensors(self, sample):
        """See prepare_update for descriptions"""
        dev = self.device
        states, actions, next_states, rewards, punishments, non_terms = list(zip(*sample))
        states = torch.cat(states)
        actions = torch.tensor(actions).long().to(dev)
        next_states = torch.cat(next_states)
        rewards = torch.tensor(rewards).long().to(dev)
        punishments = torch.tensor(punishments).long().to(dev)
        non_terms = torch.tensor(non_terms).to(dev)
        curr_mask = F.one_hot(actions, self.num_actions)
        return states, actions, next_states, rewards, punishments, non_terms, curr_mask

    def prepare_update(self, states, actions, next_states, rewards,
                       non_terms, curr_mask,
                      target, policy, weight, scale):
        """Uses target and policy network to prepare deep split-q update
            The update rule is not intuitive, see TDS or paper for info
            
            :param states: initial states for state transition
            :param next_states: end states for state transition
            :param rewards: reward gained from state transition
            :param non_terms: boolean for whether a transition ended a game
            :curr_mask: one-hot version of actions
            :next_mask: ONES vector for the next action mask
            :target: target network
            :policy: policy network
            :weight: individual reward / punishment weighting
            :scale: "fuzzy" memory factor
            :returns: end state q-values (chosen by policy and calculated by target)
                and expected_Q_values (calculated by policy with actual action taken)
        """
        dev = self.device

        next_mask = torch.ones((actions.size(0), self.num_actions))
        next_Q_vals = policy(next_states.to(dev), next_mask.to(dev))
        next_actions = next_Q_vals.max(1)[1].to(dev)
        next_mask = F.one_hot(next_actions, self.num_actions).to(dev)
        next_Q_vals = target(next_states.to(dev), next_mask)
        curr_Q_vals = target(states.to(dev), curr_mask.to(dev))
        next_Q_vals = next_Q_vals.gather(-1, next_actions.unsqueeze(1)).squeeze(-1) * non_terms
        curr_Q_vals = curr_Q_vals.gather(-1, actions.unsqueeze(1)).squeeze(-1) * non_terms
        next_Q_vals = (next_Q_vals * self.gamma) + (weight * rewards.to(dev))
        target_Q_vals = next_Q_vals - (curr_Q_vals * scale)

        expected_Q_vals = policy(states.to(dev), curr_mask.to(dev))
        expected_Q_vals = expected_Q_vals.gather(-1, actions.unsqueeze(1)).squeeze(-1)

        return target_Q_vals, expected_Q_vals

    def step_training(self, opt, policy, target, expected):
        opt.zero_grad()
        loss = self.loss(expected, target)
        loss.backward()
        for param in policy.parameters():
            param.grad.data.clamp_(-1,1)
        opt.step()
        return loss
    
    def fit_buffer_test(self, sample):
        rew_p = self.reward_pol.train()
        rew_t = self.reward_tar.eval()
        reward_scale = 1 - self.lambda_r
        punish_scale = 1 - self.lambda_p
        pun_p = self.punish_pol.train()
        pun_t = self.punish_tar.eval()
        dev = self.device

        #Unpack batch
        states, actions, next_states, rewards, punishments, non_terms, curr_mask = self.load_tensors(sample)

        #process rewards
        target_reward, expected_reward = self.prepare_update(states, actions, next_states, rewards,
                                                non_terms, curr_mask,
                                                rew_t, rew_p, self.w_r, reward_scale)

        #training step
        r_loss = self.step_training(self.opt_r, rew_p, target_reward, expected_reward)


        #process punishments
        target_punishment, expected_punishment = self.prepare_update(states, actions, next_states, punishments,
                                                    non_terms, curr_mask,
                                                    pun_t, pun_p, self.w_p, punish_scale)
        #training step
        p_loss = self.step_training(self.opt_p, pun_p, target_punishment, expected_punishment)

        return r_loss.item(), p_loss.item()

    def fit_buffer(self, sample):
        rew_p = self.reward_pol.train()
        rew_t = self.reward_tar.eval()
        reward_scale = 1 - self.lambda_r
        punish_scale = 1 - self.lambda_p
        pun_p = self.punish_pol.train()
        pun_t = self.punish_tar.eval()
        dev = self.device

        #Unpack batch
        states, actions, next_states, rewards, punishments, non_terms = list(zip(*sample))
        states = torch.cat(states).to(dev)
        actions = torch.tensor(actions).long().to(dev)
        next_states = torch.cat(next_states).to(dev)
        rewards = torch.tensor(rewards).long().to(dev)
        punishments = torch.tensor(punishments).long().to(dev)
        non_terms = torch.tensor(non_terms).to(dev)
        one_hot = torch.ones((actions.size(0), self.num_actions)).to(dev)
        curr_mask = F.one_hot(actions, self.num_actions).to(dev)

        #process rewards
        #Get best actions for next state from policy network
        next_Qr_vals = rew_p(next_states, one_hot)
        next_acts = next_Qr_vals.max(1)[1].to(dev)
        next_mask = F.one_hot(next_acts, self.num_actions).to(dev)
        #Use target network to calculate Y
        next_Qr_vals = rew_t(next_states, next_mask)
        curr_Qr_vals = rew_t(states,curr_mask)
        next_Qr_vals = next_Qr_vals.gather(-1, next_acts.unsqueeze(1)).squeeze(-1) * non_terms
        curr_Qr_vals = curr_Qr_vals.gather(-1, actions.unsqueeze(1)).squeeze(-1) * non_terms
        next_Qr_vals = (next_Qr_vals * self.gamma) + (self.w_r * rewards)
        target_Qr =  next_Qr_vals - (curr_Qr_vals * reward_scale)

        #Use policy network to calculate Q(s,a)
        expected_Qr_vals = rew_p(states, curr_mask)
        expected_Qr_vals = expected_Qr_vals.gather(-1, actions.unsqueeze(1)).squeeze(1)

        #training step
        self.opt_r.zero_grad()
        r_loss = self.loss(expected_Qr_vals, target_Qr)
        r_loss.backward()
        for param in rew_p.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt_r.step()

        #process punishments
        next_Qp_vals = pun_p(next_states, one_hot)
        next_acts = next_Qp_vals.max(1)[1].to(dev)
        next_mask = F.one_hot(next_acts, self.num_actions).to(dev)
        next_Qp_vals = pun_t(next_states, next_mask)
        curr_Qp_vals = pun_t(states,curr_mask)
        next_Qp_vals = next_Qp_vals.gather(-1, next_acts.unsqueeze(1)).squeeze(-1) * non_terms
        curr_Qp_vals = curr_Qp_vals.gather(-1, actions.unsqueeze(1)).squeeze(-1) * non_terms
        next_Qp_vals = (next_Qp_vals * self.gamma) + (self.w_p * punishments)
        target_Qp = next_Qp_vals - (curr_Qp_vals * punish_scale)

        expected_Qp_vals = pun_p(states, curr_mask)
        expected_Qp_vals = expected_Qp_vals.gather(-1, actions.unsqueeze(1)).squeeze(1)

        self.opt_p.zero_grad()
        p_loss = self.loss(expected_Qp_vals, target_Qp)
        p_loss.backward()
        for param in pun_p.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt_p.step()

        return r_loss.item(), p_loss.item()

    def get_epsilon_for_iteration(self, iteration):
        #TODO provide scaling as parameter
        return max(.05, 1-(iteration*.99/800000))

    def infer_action(self, state):
        """Chooses the max of reward + punishment"""
        pun = self.punish_pol.eval()
        rew = self.reward_pol.eval()
        dev = self.device
        state = state.unsqueeze(0).to(dev)
        mask = torch.ones(1,self.num_actions).to(dev).squeeze(0)
        r_Q = rew(state, mask)
        p_Q = pun(state, mask)
        actions = (r_Q + p_Q).squeeze(0)
        return int(actions.max(0)[1])

    def q_iteration(self, state, iteration):
        # Choose epsilon based on the iteration
        env = self.env
        epsilon = self.get_epsilon_for_iteration(iteration)

        # Choose the action
        if random.random() < epsilon:
            action = self.env.sample()
        else:
            action = self.infer_action(state)
        new_state, score, terminal, lives, frame_number = self.env.step(action)
        reward, punishment = self.reward_fcn(score, lives, self.lives, terminal, **self.reward_kwargs)
        self.lives = lives

        #handle non-terms for masking later
        non_term = int(not terminal)
        if new_state.shape[0] < 4:
            print(new_state, new_state.shape[0], score, terminal, lives, frame_number)

        #push to buffer
        mem = (state.unsqueeze(0), action,
               new_state.unsqueeze(0), reward, punishment, non_term)
        self.memory.push(mem)

        r_loss = None
        p_loss = None
        # Sample and fit
        if iteration > 64:
            batch = self.memory.sample(self.batch_size)
            r_loss, p_loss = self.fit_buffer(batch)

        return new_state, reward, punishment, score, terminal, r_loss, p_loss

    def train(self, epochs=10000, start_iter=0, updates=500):
        """Main training loop, saves statistics to logger during training
            :param epochs: number of games to play
            :param start_iter: starting iteration (default 0)
            :param updates: number of epochs before updating target model
        """
        self.updates = updates
        self.logger = Logger()
        iteration = start_iter
        running_stats = self.logger.init_stats()
        for e in range(epochs):
            terminal = False
            state, reward, terminal, lives, frames = self.env.reset()
            running_stats = self.logger.init_epoch(running_stats)
            while not terminal:
                new_state, reward, punishment, score, terminal, r_loss, p_loss = self.q_iteration(state,
                                                                                          iteration)
                iteration += 1
                if r_loss:
                    loss = (r_loss + p_loss)/2
                else:
                    loss = None
                running_stats = self.logger.update_running_stats(running_stats, score, loss)
            running_stats = self.logger.end_epoch(running_stats)
            if e%100 == 0:
                eps = self.get_epsilon_for_iteration(iteration)
                stop = self.logger.update_overall_stats(running_stats, eps, e, epochs)
                running_stats = self.logger.init_stats()
            if e%updates == 0 and e > 0:
                torch.save(self.reward_pol.state_dict(), self.r_path)
                self.load_reward_t(self.r_path)
                torch.save(self.punish_pol.state_dict(), self.p_path)
                self.load_punish_t(self.p_path)

    def load_reward_t(self, path):
        self.reward_tar.load_state_dict(torch.load(path))
        self.reward_tar.eval()

    def load_punish_t(self, path):
        self.punish_tar.load_state_dict(torch.load(path))
        self.punish_tar.eval()

    def load(self):
        self.reward_tar.load_state_dict(torch.load(self.r_path))
        self.reward_pol.load_state_dict(torch.load(self.r_path))
        self.punish_tar.load_state_dict(torch.load(self.p_path))
        self.punish_pol.load_state_dict(torch.load(self.p_path))
        self.reward_tar.eval()
        self.reward_pol.train()
        self.punish_tar.eval()
        self.punish_pol.train()

    def load_eval(self):
        self.reward_pol.load_state_dict(torch.load(self.r_path))
        self.punish_pol.load_state_dict(torch.load(self.p_path))
        
    def eval(self):
        state, reward, terminal, lives, frames = self.env.reset()
        im = plt.imshow(self.env.render())
        plt.ion()
        while not terminal:
            action = self.infer_action(state)
            new_state, reward, terminal, lives, frames = self.env.step(action, render=True, im=im)
            state = new_state
        plt.show()

    def advance(self, state, env, im):
        """For use with a ModelCombiner class"""
        action = self.infer_action(state)
        return env.step(action, render=True, im=im)
        